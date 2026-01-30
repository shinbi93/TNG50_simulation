import requests, h5py
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) API 기본 세팅
# =========================
baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key": "31e086883ec251e9fbd209d4498f9dfd"}  # 한국어: 네 API 키로 교체

def get(url, params=None, save_as=None):
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()

    if r.headers.get("content-type","").startswith("application/json"):
        return r.json()

    if save_as is None and "content-disposition" in r.headers:
        save_as = r.headers["content-disposition"].split("filename=")[-1].strip('"')
    if save_as is None:
        save_as = "download.bin"

    with open(save_as, "wb") as f:
        f.write(r.content)
    return save_as


# =========================
# 1) 대상 선택
# =========================
SIM = "TNG50-1"
SNAP = 99
SID  = 96762  # 한국어: 여기만 63864로 바꾸면 다른 은하로 동일 분석

sim  = get(baseUrl + f"{SIM}/")
snap = get(sim["snapshots"] + f"{SNAP}/")
sub  = get(snap["subhalos"] + f"{SID}/")

print("Using subhalo:", SID)
print("mass_stars:", sub.get("mass_stars"), "sfr:", sub.get("sfr"), "halfmassrad_stars:", sub.get("halfmassrad_stars"))


# =========================
# 2) stars cutout 다운로드
# =========================
cutout_request = {
    "stars": "Coordinates,Masses,GFM_InitialMass,GFM_StellarFormationTime"
}
sub_url = snap["subhalos"] + f"{SID}/"   # 한국어: subhalo 상세 endpoint를 직접 구성
fname = get(sub_url + "cutout.hdf5",
            params=cutout_request,
            save_as=f"sub{SID}_stars.hdf5")

print("saved:", fname)


# =========================
# 3) 나이(age[Gyr]) 계산: a_form -> cosmic time
# =========================
try:
    from astropy.cosmology import FlatLambdaCDM
except ModuleNotFoundError:
    raise ModuleNotFoundError("astropy가 필요합니다. 터미널에서: pip install astropy")

# 한국어: TNG의 대표 cosmology(정밀 원하면 헤더에서 다시 읽어도 됨)
h = 0.6774
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Tcmb0=2.725)

def age_gyr_from_aform(a_form):
    z_form = 1.0/np.clip(a_form, 1e-8, 1.0) - 1.0
    t0 = cosmo.age(0).value
    tf = cosmo.age(z_form).value
    return t0 - tf  # Gyr


# =========================
# 4) 단위 변환 + 반지름 계산
# =========================
def to_physical_kpc(coord_ckpc_h, a=1.0, h=0.6774):
    # 한국어: ckpc/h -> physical kpc
    return coord_ckpc_h * a / h

def to_physical_msun(m_1e10_msun_h, h=0.6774):
    # 한국어: (1e10 Msun/h) -> Msun
    return m_1e10_msun_h * 1e10 / h


# =========================
# 5) IMF + MS 생존 개수(근사) 모델
# =========================
def kroupa_xi(m):
    # 한국어: Kroupa IMF(unnormalized)
    m = np.asarray(m)
    xi = np.zeros_like(m, dtype=float)
    a1, a2 = 1.3, 2.3
    xi[m < 0.5] = m[m < 0.5] ** (-a1)
    xi[m >= 0.5] = (0.5 ** (a2 - a1)) * m[m >= 0.5] ** (-a2)
    return xi

def integrate_number(m1, m2, ngrid=2000):
    ms = np.logspace(np.log10(m1), np.log10(m2), ngrid)
    return np.trapz(kroupa_xi(ms), ms)

def integrate_mass(m1, m2, ngrid=2000):
    ms = np.logspace(np.log10(m1), np.log10(m2), ngrid)
    return np.trapz(ms * kroupa_xi(ms), ms)

IMF_MMIN, IMF_MMAX = 0.08, 100.0
IMF_MASS_NORM = integrate_mass(IMF_MMIN, IMF_MMAX)

def turnoff_mass_from_age_gyr(age_gyr):
    # 한국어: t_MS ≈ 10 Gyr (m/Msun)^(-2.5) => m_to ≈ (10/t)^(0.4)
    age_gyr = np.maximum(age_gyr, 1e-3)
    return (10.0 / age_gyr) ** 0.4

def count_ms_stars(M_init_msun, age_gyr, m1, m2):
    m_to = turnoff_mass_from_age_gyr(age_gyr)
    upper = min(m2, m_to)
    if upper <= m1:
        return 0.0
    return M_init_msun * integrate_number(m1, upper) / IMF_MASS_NORM


# =========================
# 6) 프로파일 계산
# =========================
m_bins = [(0.1,0.3),(0.3,0.8),(0.8,1.2),(1.2,2.0)]
Rmax_kpc = 30.0
nbins = 30

with h5py.File(fname, "r") as f:
    pt = f["PartType4"]
    coords = pt["Coordinates"][:]                 # ckpc/h
    m_init = pt["GFM_InitialMass"][:]             # 1e10 Msun/h
    a_form = pt["GFM_StellarFormationTime"][:]    # scale factor

# 한국어: subhalo 중심 좌표 키가 환경마다 달라서 안전하게 찾기
cen_key_candidates = ["pos", "SubhaloPos", "center", "cm"]
cen = None
for k in cen_key_candidates:
    if k in sub:
        cen = np.array(sub[k], dtype=float)
        print(f"Using center from sub['{k}']")
        break

if cen is None:
    # 한국어: sub JSON에 중심 좌표가 없으면, cutout Header에서 SubhaloPos를 찾는다
    with h5py.File(fname, "r") as f:
        if "Header" in f and "SubhaloPos" in f["Header"].attrs:
            cen = np.array(f["Header"].attrs["SubhaloPos"], dtype=float)
            print("Using center from cutout Header attrs['SubhaloPos']")
        else:
            # 한국어: 최후 수단: 좌표의 질량가중 중심(근사)
            pt = h5py.File(fname, "r")["PartType4"]
            coords_tmp = pt["Coordinates"][:]
            m_tmp = pt["GFM_InitialMass"][:]
            cen = np.average(coords_tmp, axis=0, weights=m_tmp)
            print("Using mass-weighted center from star particles (fallback)")


# 단위 변환
xyz = to_physical_kpc(coords - cen[None,:], a=1.0, h=h)   # kpc
R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)                    # 원통반지름(정렬 전 1차 버전)

M_init_msun = to_physical_msun(m_init, h=h)
age_gyr = age_gyr_from_aform(a_form)

edges = np.linspace(0.0, Rmax_kpc, nbins+1)
centers = 0.5*(edges[:-1] + edges[1:])
inds = np.digitize(R, edges) - 1
valid = (inds >= 0) & (inds < nbins)

profile = np.zeros((nbins, len(m_bins)))

idxs = np.where(valid)[0]
for i in idxs:
    b = inds[i]
    for j,(m1,m2) in enumerate(m_bins):
        profile[b, j] += count_ms_stars(M_init_msun[i], age_gyr[i], m1, m2)

# =========================
# 7) 플롯
# =========================
for j,(m1,m2) in enumerate(m_bins):
    plt.plot(centers, profile[:,j], label=f"{m1}-{m2} Msun")

plt.yscale("log")
plt.xlabel("R [kpc]")
plt.ylabel("N_MS per radial bin (arbitrary norm)")
plt.title(f"TNG50-1 snap{SNAP} sub{SID}: MS star counts vs R")
plt.legend()
plt.tight_layout()
plt.show()
