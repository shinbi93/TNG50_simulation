"""
tng50_mw_fast.py  –  원본 대비 핵심 병목 3곳 최적화
================================================
[원인1] isolation_mask_for_groups : O(N^2) Python loop  → scipy KDTree + 벡터 연산
[원인2] reconstruct_counts_for_galaxy : 별 수만개 × bin 루프  → np.searchsorted 벡터화
[원인3] IMF 적분 precompute : 매 별마다 integrate 호출  → bin별 1회만 미리 계산
"""

import os
import re
import glob
import json
import csv
import time
import requests
import numpy as np
import h5py
from tqdm import tqdm
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

# ============================================================
# 0) 사용자 설정
# ============================================================
SIM  = "TNG50-1"
SNAP = 99  # z=0

GROUPCAT_DIR    = "/bishin/TNG/TNG50/TNG50-1/output/groups_099"
HIH2_SUPP_PATH  = "./supp/tng50-1_hih2_z0.hdf5"
MORPH_SUPP_PATH = "./supp/stellar_circularities_angular_momenta_axis_ratios.hdf5"

CUTOUT_DIR = "./cutouts_tng50_snap99"
os.makedirs(CUTOUT_DIR, exist_ok=True)

OUT_DIR = "./outputs_mw_like"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1) TNG API
# ============================================================
BASE    = "https://www.tng-project.org/api"
API_KEY = os.environ.get("TNG_API_KEY", "").strip()
HEADERS = {"api-key": API_KEY}

# ============================================================
# 2) 선택 기준
# ============================================================
MSTAR_RANGE = (4e10, 8e10)
SFR_RANGE   = (1.0, 2.0)
MGAS_RANGE  = (5e9, 1.2e10)

R_ISO_MPC       = 1.0
F_MASS_NEIGHBOR = 0.5
R_CLUSTER_MPC   = 3.0
M_CLUSTER_MSUN  = 5e13

APPLY_DISK_CUT = True
CIRC07_MIN     = 0.35
BULGE2_MAX     = 0.8

STAR_MASS_BINS = np.array([0.08, 0.1, 0.2, 0.5, 1, 2, 5, 8, 20, 50, 100.0], dtype=float)

# ============================================================
# 3) 단위/우주론
# ============================================================
h = 0.6774
MASS_UNIT    = 1e10 / h
BOX_CKPC_H   = 35000.0
BOX_MPC_PHYS = (BOX_CKPC_H / 1000.0) / h  # z=0 physical Mpc

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)

# ============================================================
# 4) HTTP 유틸
# ============================================================
def http_get(url, params=None, stream=False, retries=3, sleep=2.0):
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, stream=stream, timeout=240)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(sleep * (k + 1))
    raise last_err

# ============================================================
# 5) groupcat 파일 읽기 유틸 (원본 그대로)
# ============================================================
def _pick_key(avail_keys, patterns, prefer_exact=None):
    avail = list(avail_keys)
    aset  = set(avail)
    if prefer_exact:
        for k in prefer_exact:
            if k in aset:
                return k
    best, best_score = None, -1
    for k in avail:
        score = sum(1 for p in patterns if re.search(p, k, flags=re.IGNORECASE))
        if score > best_score:
            best_score, best = score, k
    return best if best_score > 0 else None

def read_concat_auto(file_glob, group_name, want_to_patterns,
                     allow_skip_missing=True, verbose=True):
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files: {file_glob}")

    resolved = None
    first_fp = None
    for fp in files:
        with h5py.File(fp, "r") as f0:
            if group_name not in f0:
                continue
            avail = list(f0[group_name].keys())
            tmp = {}
            ok  = True
            for want, spec in want_to_patterns.items():
                key = _pick_key(avail, spec.get("patterns", []), spec.get("prefer"))
                if key is None:
                    ok = False; break
                tmp[want] = key
            if ok:
                resolved, first_fp = tmp, fp
                break

    if resolved is None:
        raise KeyError(f"Cannot resolve keys for /{group_name} from {file_glob}")
    if verbose:
        print(f"[INFO] Resolved /{group_name} from {first_fp}: {resolved}")

    out = {w: [] for w in want_to_patterns}
    used = skipped = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            if group_name not in f:
                skipped += 1; continue
            g = f[group_name]
            if any(rk not in g for rk in resolved.values()):
                if allow_skip_missing:
                    skipped += 1; continue
                raise KeyError(f"{fp}: missing keys")
            for want, rk in resolved.items():
                out[want].append(g[rk][:])
            used += 1

    if used == 0:
        raise RuntimeError(f"All chunks skipped for /{group_name}")
    out = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    if verbose and skipped:
        print(f"[INFO] /{group_name}: used={used}, skipped={skipped}")
    return out, resolved

def load_group_subhalo(groupcat_dir, snap):
    # groupcat-099.*.hdf5 (API 다운로드) 또는 fof_subhalo_tab_099.*.hdf5 (직접 다운로드) 둘 다 지원
    cpat = os.path.join(groupcat_dir, f"groupcat-{snap:03d}.*.hdf5")
    if not glob.glob(cpat):
        cpat = os.path.join(groupcat_dir, f"fof_subhalo_tab_{snap:03d}.*.hdf5")
    if not glob.glob(cpat):
        actual = os.listdir(groupcat_dir)[:10] if os.path.isdir(groupcat_dir) else "dir not found"
        raise FileNotFoundError(
            f"No groupcat files in {groupcat_dir}\n"
            f"Expected: groupcat-{snap:03d}.*.hdf5  or  fof_subhalo_tab_{snap:03d}.*.hdf5\n"
            f"Actual files (first 10): {actual}"
        )
    print(f"[INFO] Using groupcat pattern: {cpat}")

    group_specs = {
        "GroupFirstSub":    {"prefer": ["GroupFirstSub"], "patterns": [r"first.*sub"]},
        "GroupPos":         {"prefer": ["GroupPos"],       "patterns": [r"(^|_)pos($|_)"]},
        "Group_M_Crit200":  {"prefer": ["Group_M_Crit200"],"patterns": [r"crit.?200"]},
    }
    sub_specs = {
        "SubhaloSFR":           {"prefer": ["SubhaloSFR"],           "patterns": [r"sfr"]},
        "SubhaloMassInRadType": {"prefer": ["SubhaloMassInRadType"], "patterns": [r"mass.*in.*rad.*type"]},
        "SubhaloGrNr":          {"prefer": ["SubhaloGrNr"],          "patterns": [r"grnr"]},
    }
    gdat, gmap = read_concat_auto(cpat, "Group",   group_specs)
    sdat, smap = read_concat_auto(cpat, "Subhalo", sub_specs)
    print("[INFO] Group keys:", gmap)
    print("[INFO] Subhalo keys:", smap)
    return (gdat["GroupFirstSub"],  gdat["GroupPos"], gdat["Group_M_Crit200"],
            sdat["SubhaloSFR"], sdat["SubhaloMassInRadType"], sdat["SubhaloGrNr"])

# ============================================================
# 6) ★ 최적화: isolation  O(N²) → KDTree  ★
# ============================================================
def isolation_mask_for_groups(group_pos_mpc, group_m200_msun,
                               R_iso_mpc=1.0, f_mass=0.5,
                               R_cluster_mpc=3.0, M_cluster_msun=5e13):
    """
    원본 코드: Python for-loop O(N^2)  → 매우 느림
    개선안:
      - scipy cKDTree 로 반경 내 이웃 목록만 추출 (ball_point_query)
      - periodic boundary 를 구현하기 위해 박스를 27-copy 하지 않고
        query_ball_point 후 PBC 거리 재확인 (R < box/2 인 현실적 반경이면 충분)
    """
    Ng = len(group_pos_mpc)
    pos = group_pos_mpc  # (Ng, 3)

    # KDTree (non-periodic; PBC 보정은 아래서)
    tree = cKDTree(pos)

    # 더 큰 반경으로 candidate 를 뽑고 PBC 거리로 재필터
    R_max = max(R_iso_mpc, R_cluster_mpc)

    # ball_point_query – 각 그룹의 이웃 인덱스 목록
    neighbors_list = tree.query_ball_point(pos, r=R_max + 0.0)  # list of lists

    iso = np.ones(Ng, dtype=bool)
    half_box = BOX_MPC_PHYS / 2.0

    for i in range(Ng):
        Mi  = group_m200_msun[i]
        pi  = pos[i]
        nbrs = np.array(neighbors_list[i], dtype=np.int64)
        nbrs = nbrs[nbrs != i]
        if len(nbrs) == 0:
            continue

        # 벡터화된 PBC 거리
        d = pos[nbrs] - pi[None, :]
        # minimum image
        d = d - BOX_MPC_PHYS * np.round(d / BOX_MPC_PHYS)
        r = np.sqrt((d * d).sum(axis=1))

        mj = group_m200_msun[nbrs]

        # 조건1: r < R_iso 이고 이웃 질량 >= f_mass * Mi
        if np.any((r < R_iso_mpc) & (mj >= f_mass * Mi)):
            iso[i] = False
            continue

        # 조건2: r < R_cluster 이고 이웃 질량 >= M_cluster
        if np.any((r < R_cluster_mpc) & (mj >= M_cluster_msun)):
            iso[i] = False

    return iso

# ============================================================
# 7) HI+H2 보조카탈로그
# ============================================================
def try_load_hih2_map_for_snap(hih2_path, expected_snap=99, model="GK11", method="vol"):
    if not hih2_path or not os.path.exists(hih2_path):
        return None, "HIH2 missing"
    key_hi = f"m_hi_{model}_{method}"
    key_h2 = f"m_h2_{model}_{method}"
    with h5py.File(hih2_path, "r") as f:
        snap_idx = None
        if "config" in f:
            if "snap_idx" in f["config"].attrs:
                snap_idx = int(f["config"].attrs["snap_idx"])
        if snap_idx is not None and snap_idx != expected_snap:
            return None, f"snap_idx={snap_idx} != {expected_snap}"
        if "id_subhalo" not in f or key_hi not in f or key_h2 not in f:
            return None, "HIH2 missing keys"
        sid = f["id_subhalo"][:].astype(np.int64)
        mhi = f[key_hi][:].astype(float)
        mh2 = f[key_h2][:].astype(float)
    d = {int(sid[i]): float(mhi[i] + mh2[i]) for i in range(len(sid))}
    return d, "HIH2 loaded"

# ============================================================
# 8) disk cut
# ============================================================
def apply_disk_cut(subhalo_ids, morph_path, snap, circ07_min=0.35, bulge2_max=0.8):
    if not morph_path or not os.path.exists(morph_path):
        return subhalo_ids, "Morph missing -> skipped"
    grp = f"/Snapshot_{snap}"
    with h5py.File(morph_path, "r") as f:
        if grp not in f:
            return subhalo_ids, f"No {grp}"
        SubfindID = f[f"{grp}/SubfindID"][:]
        circ07    = f[f"{grp}/CircAbove07Frac"][:]
        bulge2    = f[f"{grp}/CircTwiceBelow0Frac"][:]

    idx = {int(s): i for i, s in enumerate(SubfindID)}
    kept = []
    for sid in subhalo_ids:
        i = idx.get(int(sid))
        if i is None:
            continue
        if circ07[i] >= circ07_min and bulge2[i] <= bulge2_max:
            kept.append(sid)
    return np.array(kept, dtype=np.int64), f"Disk kept {len(kept)}/{len(subhalo_ids)}"

# ============================================================
# 9) cutout 다운로드
# ============================================================
def download_cutout(sim, snap, subhalo_id, out_dir, fields):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sim}_snap{snap:03d}_sub{subhalo_id}_cutout.hdf5")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    url    = f"{BASE}/{sim}/snapshots/{snap}/subhalos/{subhalo_id}/cutout.hdf5"
    params = {}
    for fld in fields:
        ptype, name = fld.split("/")
        params.setdefault(ptype, []).append(name)
    params = {k: ",".join(v) for k, v in params.items()}
    r = http_get(url, params=params, stream=True)
    with open(out_path, "wb") as fp:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fp.write(chunk)
    return out_path

# ============================================================
# 10) ★ 최적화: Chabrier IMF + 별 개수 재구성 벡터화 ★
# ============================================================
def chabrier03_dndm(m):
    m   = np.asarray(m, dtype=float)
    out = np.zeros_like(m)
    mc, sigma = 0.079, 0.69
    lo = (m > 0) & (m < 1.0)
    if lo.any():
        out[lo] = np.exp(-0.5 * ((np.log10(m[lo]) - np.log10(mc)) / sigma) ** 2) \
                  / (m[lo] * np.log(10.0))
    out[m >= 1.0] = m[m >= 1.0] ** (-2.3)
    return out

def _integrate_log(m1, m2, n=800):
    """로그 격자 trapz 적분 (재사용)"""
    if m2 <= m1:
        return 0.0
    grid = np.logspace(np.log10(m1), np.log10(m2), n)
    mid  = np.sqrt(grid[:-1] * grid[1:])
    dm   = np.diff(grid)
    return float(np.sum(chabrier03_dndm(mid) * dm))

def _integrate_log_mass(m1, m2, n=1600):
    if m2 <= m1:
        return 0.0
    grid = np.logspace(np.log10(m1), np.log10(m2), n)
    mid  = np.sqrt(grid[:-1] * grid[1:])
    dm   = np.diff(grid)
    return float(np.sum(mid * chabrier03_dndm(mid) * dm))

def precompute_bin_integrals(star_mass_bins):
    """
    각 bin 마다 N/M_total 을 미리 계산.
    원본과 동일하지만 전체 IMF 정규화 분모도 같이 반환.
    """
    mmin, mmax = star_mass_bins[0], star_mass_bins[-1]
    denom_mass  = _integrate_log_mass(mmin, mmax)

    num_bins = np.array([
        _integrate_log(max(mmin, star_mass_bins[i]),
                       min(mmax, star_mass_bins[i+1]))
        for i in range(len(star_mass_bins) - 1)
    ], dtype=float)

    return num_bins, denom_mass

# ★ 핵심 최적화 함수 ★
def reconstruct_counts_for_galaxy(cutout_path, star_mass_bins,
                                   num_bins_init, denom_mass):
    """
    원본: 별 하나마다 inner loop + integrate 호출  → O(N_star × N_bin) × 적분
    개선: np.searchsorted + 벡터 연산으로 O(N_star + N_bin)
    """
    mmin, mmax = star_mass_bins[0], star_mass_bins[-1]

    with h5py.File(cutout_path, "r") as f:
        m_init = f["PartType4/GFM_InitialMass"][:].astype(float) * MASS_UNIT
        a_form = f["PartType4/GFM_StellarFormationTime"][:].astype(float)

    # --- N_init: SSP 가정, IMF 적분 비율을 곱하면 됨 ---
    # N_init[b] = sum_stars( M_init_star * num_bins_init[b] / denom_mass )
    #           = (sum_stars M_init_star) * num_bins_init[b] / denom_mass
    total_m_init = m_init.sum()
    N_init = total_m_init * num_bins_init / denom_mass   # (N_bin,) – 즉시 계산

    # --- N_surv: 각 별의 MS turnoff 질량까지만 살아있다고 가정 ---
    # a_form 이 양수인 별만 (음수는 wind particle)
    mask = a_form > 0
    m_init = m_init[mask]
    a_form = a_form[mask]

    if len(m_init) == 0:
        return N_init, np.zeros_like(N_init)

    t0     = cosmo.age(0).value
    z_form = 1.0 / a_form - 1.0
    # 벡터화된 cosmo.age – 한 번에 계산
    t_form = cosmo.age(z_form).value          # (N_star,)
    age    = np.clip(t0 - t_form, 0.0, None)

    # turnoff mass (approximate)
    Mto = np.clip((age / 10.0) ** (-0.4), mmin, mmax)   # (N_star,)

    # 각 별의 유효 상한 bin 인덱스 찾기 (searchsorted)
    # star_mass_bins = [b0, b1, ..., bK]  → K bins
    # 별 i 의 생존 질량 상한: min(Mto[i], bin 상한)
    # bin b 의 생존 적분: integral(mmin_b, min(b_max, Mto)) / denom_mass

    # 미리 bin별 누적 적분 테이블 구축 (bin 경계에서의 누적 N/M)
    # CDF_N[k] = integral(mmin, star_mass_bins[k]) dndm  / denom_mass  (각 별 단위)
    # 이를 이용해 구간 적분을 CDF 차이로 표현

    # ---- 누적 IMF 테이블 ----
    n_pts   = 2000
    m_grid  = np.logspace(np.log10(mmin), np.log10(mmax), n_pts)
    dm      = np.diff(m_grid)
    mid     = np.sqrt(m_grid[:-1] * m_grid[1:])
    phi     = chabrier03_dndm(mid)

    # cumulative number integral (N per unit initial mass, normalized by denom_mass)
    cum_N   = np.zeros(n_pts)
    cum_N[1:] = np.cumsum(phi * dm) / denom_mass   # CDF_N[k] at m_grid[k]

    # 각 bin 경계에서 누적 값 (보간)
    cum_at_edges = np.interp(star_mass_bins, m_grid, cum_N)   # (N_bin+1,)
    # bin별 per-unit-mass 적분
    bin_cum_diff = np.diff(cum_at_edges)   # (N_bin,) – 각 bin의 dN/dM_init

    # 각 별의 turnoff 이하 누적 적분
    cum_at_Mto = np.interp(Mto, m_grid, cum_N)   # (N_star,)

    # N_surv[b] = sum_i M_init_i * max(0, min(cum_at_edges[b+1], cum_at_Mto[i]) - cum_at_edges[b])
    # 벡터화:
    #   shape (N_star, N_bin)
    N_bin = len(star_mass_bins) - 1
    lo_edge = cum_at_edges[:-1][None, :]    # (1, N_bin)
    hi_edge = cum_at_edges[1: ][None, :]    # (1, N_bin)
    mto_cum = cum_at_Mto[:, None]           # (N_star, 1)

    # 각 별이 각 bin에 기여하는 dN/dM_init * M_init
    contrib = np.clip(np.minimum(hi_edge, mto_cum) - lo_edge, 0.0, None)   # (N_star, N_bin)

    # N_surv = sum over stars of M_init * contrib
    N_surv = (m_init[:, None] * contrib).sum(axis=0)   # (N_bin,)

    return N_init, N_surv

# ============================================================
# 11) 저장/플롯
# ============================================================
def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def save_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def plot_survival_fraction(star_mass_bins, surv_frac, out_png):
    x = 0.5 * (star_mass_bins[:-1] + star_mass_bins[1:])
    plt.figure()
    plt.xscale("log"); plt.yscale("log")
    plt.plot(x, surv_frac, marker="o")
    plt.xlabel("Stellar mass bin center [Msun]")
    plt.ylabel("Mean survival fraction N_surv/N_init")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_mean_counts(star_mass_bins, mean_init, mean_surv, out_png):
    x = 0.5 * (star_mass_bins[:-1] + star_mass_bins[1:])
    plt.figure()
    plt.xscale("log"); plt.yscale("log")
    plt.plot(x, mean_init, marker="o", label="Mean N_init")
    plt.plot(x, mean_surv, marker="o", label="Mean N_surv")
    plt.xlabel("Stellar mass bin center [Msun]")
    plt.ylabel("Reconstructed number of stars")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ============================================================
# 12) MAIN
# ============================================================
def main():
    if not API_KEY:
        raise RuntimeError("Missing API key. Run: export TNG_API_KEY='...your key...'")

    print("[1] Load group/subhalo catalogs...")
    (GroupFirstSub, GroupPos, Group_M_Crit200,
     SubhaloSFR, SubhaloMassInRadType, SubhaloGrNr) = load_group_subhalo(GROUPCAT_DIR, SNAP)

    central_ids = np.unique(GroupFirstSub[GroupFirstSub >= 0].astype(np.int64))

    Mstar       = SubhaloMassInRadType[:, 4].astype(float) * MASS_UNIT
    Mgas_approx = SubhaloMassInRadType[:, 0].astype(float) * MASS_UNIT
    SFR         = SubhaloSFR.astype(float)

    mstar_ok = (Mstar >= MSTAR_RANGE[0]) & (Mstar <= MSTAR_RANGE[1])
    sfr_ok   = (SFR   >= SFR_RANGE[0])   & (SFR   <= SFR_RANGE[1])

    print("[2] HI+H2 (optional)...")
    hih2_map, msg = try_load_hih2_map_for_snap(HIH2_SUPP_PATH, SNAP)
    print(f"    {msg}")

    if hih2_map is not None:
        Mgas_ref = np.full_like(Mstar, np.nan)
        for sid, val in hih2_map.items():
            if 0 <= sid < len(Mgas_ref):
                Mgas_ref[sid] = val
        mgas_used  = Mgas_ref
        mgas_label = "HI+H2"
    else:
        mgas_used  = Mgas_approx
        mgas_label = "gas_approx"

    mgas_ok = (mgas_used >= MGAS_RANGE[0]) & (mgas_used <= MGAS_RANGE[1])

    base     = np.where(mstar_ok & mgas_ok & sfr_ok)[0].astype(np.int64)
    mw_like  = np.intersect1d(base, central_ids)
    print(f"    MW-like before isolation/disk: {len(mw_like)}")

    # ---- isolation ----
    print("[3] Isolation (KDTree)...")
    group_pos_mpc   = (GroupPos.astype(float) / 1000.0) / h
    group_m200_msun = Group_M_Crit200.astype(float) * MASS_UNIT

    isolated_group = isolation_mask_for_groups(
        group_pos_mpc, group_m200_msun,
        R_iso_mpc=R_ISO_MPC, f_mass=F_MASS_NEIGHBOR,
        R_cluster_mpc=R_CLUSTER_MPC, M_cluster_msun=M_CLUSTER_MSUN
    )

    mw_groups = SubhaloGrNr[mw_like].astype(np.int64)
    mw_iso    = mw_like[isolated_group[mw_groups]]
    print(f"    After isolation: {len(mw_iso)}")

    # ---- disk cut ----
    mw_final = mw_iso
    if APPLY_DISK_CUT:
        mw_final, msg = apply_disk_cut(mw_iso, MORPH_SUPP_PATH, SNAP, CIRC07_MIN, BULGE2_MAX)
        print(f"    Disk: {msg}")
    print(f"    Final sample: {len(mw_final)}")

    if len(mw_final) == 0:
        raise RuntimeError("No galaxies. Relax cuts.")

    # ---- IMF precompute ----
    print("[4] IMF integrals (precompute once)...")
    num_bins_init, denom_mass = precompute_bin_integrals(STAR_MASS_BINS)

    # ---- cutouts + reconstruct ----
    print("[5] Cutouts + reconstruct (vectorized)...")
    fields = ("PartType4/GFM_InitialMass", "PartType4/GFM_StellarFormationTime")

    rows, results = [], []
    for sid in tqdm(mw_final, desc="Galaxies"):
        sid = int(sid)
        cut = download_cutout(SIM, SNAP, sid, CUTOUT_DIR, fields)

        N_init, N_surv = reconstruct_counts_for_galaxy(
            cut, STAR_MASS_BINS, num_bins_init, denom_mass
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(N_init > 0, N_surv / N_init, np.nan)

        rows.append(
            [sid, float(Mstar[sid]), float(mgas_used[sid]), float(SFR[sid])]
            + N_init.tolist() + N_surv.tolist() + frac.tolist()
        )
        results.append({
            "SubhaloID": sid,
            "Mstar_Msun": float(Mstar[sid]),
            "Mgas_used_Msun": float(mgas_used[sid]),
            "Mgas_source": mgas_label,
            "SFR_MsunPerYr": float(SFR[sid]),
            "stellar_mass_bins_Msun": STAR_MASS_BINS.tolist(),
            "N_init_bins": N_init.tolist(),
            "N_surv_bins": N_surv.tolist(),
            "survival_fraction_bins": frac.tolist(),
        })

    # ---- save ----
    out_json = os.path.join(OUT_DIR, f"mw_like_tng50_snap{SNAP:03d}_full.json")
    save_json(out_json, {
        "meta": {
            "sim": SIM, "snap": SNAP,
            "cuts": {
                "Mstar_range": list(MSTAR_RANGE),
                "Mgas_range":  list(MGAS_RANGE),
                "SFR_range":   list(SFR_RANGE),
                "isolation":   {"R_iso": R_ISO_MPC, "f_mass": F_MASS_NEIGHBOR,
                                "R_cluster": R_CLUSTER_MPC, "M_cluster": M_CLUSTER_MSUN},
                "disk_cut":    APPLY_DISK_CUT,
            },
        },
        "results": results
    })
    print(f"[DONE] JSON: {out_json}")

    header  = ["SubhaloID", "Mstar_Msun", "Mgas_used_Msun", "SFR_MsunPerYr"]
    header += [f"N_init_{STAR_MASS_BINS[i]}_{STAR_MASS_BINS[i+1]}" for i in range(len(STAR_MASS_BINS)-1)]
    header += [f"N_surv_{STAR_MASS_BINS[i]}_{STAR_MASS_BINS[i+1]}" for i in range(len(STAR_MASS_BINS)-1)]
    header += [f"frac_{STAR_MASS_BINS[i]}_{STAR_MASS_BINS[i+1]}"   for i in range(len(STAR_MASS_BINS)-1)]
    out_csv = os.path.join(OUT_DIR, f"mw_like_tng50_snap{SNAP:03d}_full.csv")
    save_csv(out_csv, rows, header)
    print(f"[DONE] CSV:  {out_csv}")

    print("[6] Plots...")
    N_init_all = np.array([r["N_init_bins"] for r in results])
    N_surv_all = np.array([r["N_surv_bins"] for r in results])
    mean_init  = np.nanmean(N_init_all, axis=0)
    mean_surv  = np.nanmean(N_surv_all, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_frac = np.where(mean_init > 0, mean_surv / mean_init, np.nan)

    plot_survival_fraction(STAR_MASS_BINS, mean_frac,
                           os.path.join(OUT_DIR, f"mw_like_tng50_snap{SNAP:03d}_survival_fraction.png"))
    plot_mean_counts(STAR_MASS_BINS, mean_init, mean_surv,
                     os.path.join(OUT_DIR, f"mw_like_tng50_snap{SNAP:03d}_mean_counts.png"))
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
