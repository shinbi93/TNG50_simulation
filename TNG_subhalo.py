import requests

baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key": "31e086883ec251e9fbd209d4498f9dfd"}

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


# ---- 2단계: TNG50-1 / snapshot 99 / subhalo top ----
sim = get(baseUrl + "TNG50-1/")
snap = get(sim["snapshots"] + "99/")

subs = get(snap["subhalos"], params={"limit": 20, "order_by": "-mass_stars"})
top = subs["results"]
ids = [x["id"] for x in top[:5]]

print("Top 5 SubhaloIDs:", ids)

# ---- (여기부터) 후보 비교 코드 붙여넣기 ----
rows = []
for sid in ids:
    s = get(snap["subhalos"] + f"{sid}/")
    rows.append({
        "id": sid,
        "mass_stars": s.get("mass_stars"),
        "sfr": s.get("sfr"),
        "grnr": s.get("grnr"),
        "halfmassrad_stars": s.get("halfmassrad_stars"),
        "vmax": s.get("vmax"),
    })

print("\n[Subhalo basic properties]")
for r in rows:
    print(r)

# ---- (교체) Central check: snap에 halos 키가 없을 때도 동작하도록 ----
halo_base = snap.get("halos", None)
if halo_base is None:
    # 한국어: 보통 스냅샷 URL 뒤에 halos/ 가 붙는다
    halo_base = snap["url"] + "halos/"

print("halo_base =", halo_base)

print("\n[Central check]")
for sid in ids:
    s = get(snap["subhalos"] + f"{sid}/")

    # 한국어: halos/가 안 되면 groups/로 바꿔서 시도하면 됨
    gr = get(halo_base + f"{s['grnr']}/")

    first_sub = gr.get("first_subhalo_id", gr.get("GroupFirstSub", None))
    is_central = (first_sub == sid)
    print("sid", sid, "grnr", s["grnr"], "first_sub", first_sub, "central?", is_central)


import h5py
import numpy as np

# 한국어: 1) GroupFirstSub 필드만 다운로드 (파일로 저장됨)
groupfirstsub_file = get(
    baseUrl + "TNG50-1/files/groupcat-99/",
    params={"Group": "GroupFirstSub"},
    save_as="groupcat-99_GroupFirstSub.hdf5"
)
print("saved:", groupfirstsub_file)

# 한국어: 2) 읽어서 central 판정
with h5py.File("groupcat-99_GroupFirstSub.hdf5", "r") as f:
    # 보통 Group/GroupFirstSub 형태인데, 혹시 다르면 keys 출력로 확인
    if "Group" in f and "GroupFirstSub" in f["Group"]:
        gfs = f["Group"]["GroupFirstSub"][:]
    else:
        # 한국어: 구조가 다를 수 있으니 전체 키 확인
        print("HDF5 keys:", list(f.keys()))
        raise KeyError("Group/GroupFirstSub를 찾지 못했어요. 위 keys 출력 보고 경로를 바꿔야 합니다.")

print("\n[Central check via GroupFirstSub]")
for sid in ids:
    s = get(snap["subhalos"] + f"{sid}/")
    grnr = s["grnr"]
    is_central = (int(gfs[grnr]) == sid)
    print("sid", sid, "grnr", grnr, "GroupFirstSub[grnr]", int(gfs[grnr]), "central?", is_central)
