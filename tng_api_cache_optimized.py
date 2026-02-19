"""
TNG API 분석 - 캐시 우선 버전

캐시 시스템:
- 한 번 다운로드한 데이터는 자동 저장
- 재실행 시 다운로드 없이 즉시 로드
- 필요시 캐시 삭제로 재다운로드 가능

사용법:
1. 처음 실행: 데이터 다운로드 + 캐시 저장
2. 다시 실행: 캐시에서 즉시 로드 (다운로드 없음)
3. 캐시 삭제: rm -rf tng_api_cache/
"""

import requests
import numpy as np
import h5py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.integrate import quad
import warnings
import sys

warnings.filterwarnings('ignore')


# =============================================================================
# 캐시 관리 클래스
# =============================================================================

class CacheManager:
    """캐시 관리 전용 클래스"""
    
    def __init__(self, cache_dir='tng_api_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 캐시 상태 파일
        self.status_file = self.cache_dir / 'cache_status.json'
        self._load_status()
    
    def _load_status(self):
        """캐시 상태 로드"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'subhalo_catalog': False,
                'group_catalog': False,
                'downloaded_galaxies': []
            }
    
    def _save_status(self):
        """캐시 상태 저장"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def is_cached(self, item_type, item_id=None):
        """캐시 존재 확인"""
        if item_type == 'subhalo_catalog':
            return self.status.get('subhalo_catalog', False)
        elif item_type == 'group_catalog':
            return self.status.get('group_catalog', False)
        elif item_type == 'galaxy':
            return item_id in self.status.get('downloaded_galaxies', [])
        return False
    
    def mark_cached(self, item_type, item_id=None):
        """캐시 완료 표시"""
        if item_type == 'subhalo_catalog':
            self.status['subhalo_catalog'] = True
        elif item_type == 'group_catalog':
            self.status['group_catalog'] = True
        elif item_type == 'galaxy':
            if item_id not in self.status.get('downloaded_galaxies', []):
                self.status['downloaded_galaxies'].append(item_id)
        
        self._save_status()
    
    def get_cache_info(self):
        """캐시 정보 반환"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        n_files = len(list(self.cache_dir.rglob('*')))
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'n_files': n_files,
            'n_galaxies': len(self.status.get('downloaded_galaxies', [])),
            'has_catalogs': self.status.get('subhalo_catalog', False) and self.status.get('group_catalog', False)
        }
    
    def print_cache_info(self):
        """캐시 정보 출력"""
        info = self.get_cache_info()
        
        print("\n" + "="*70)
        print("💾 캐시 상태")
        print("="*70)
        print(f"  📁 위치: {self.cache_dir}")
        print(f"  📊 크기: {info['total_size_mb']:.1f} MB")
        print(f"  📄 파일 수: {info['n_files']}개")
        print(f"  ✓ 카탈로그: {'있음' if info['has_catalogs'] else '없음'}")
        print(f"  ✓ 다운로드된 은하: {info['n_galaxies']}개")
        
        if info['has_catalogs']:
            print(f"\n  💡 카탈로그는 캐시에서 즉시 로드됩니다 (다운로드 없음)")
        if info['n_galaxies'] > 0:
            print(f"  💡 {info['n_galaxies']}개 은하는 캐시에서 즉시 로드됩니다")
        
        print("="*70)


# =============================================================================
# Chabrier IMF 클래스
# =============================================================================

class ChabrierIMF:
    """Chabrier (2003) IMF"""
    
    def __init__(self):
        self.mc = 0.079
        self.sigma = 0.69
        self.A = 0.158
        self.alpha = 2.3
        
    def __call__(self, mass):
        mass = np.atleast_1d(mass)
        imf = np.zeros_like(mass)
        
        mask_low = mass < 1.0
        if np.any(mask_low):
            m_low = mass[mask_low]
            imf[mask_low] = (self.A / m_low) * np.exp(
                -0.5 * (np.log10(m_low) - np.log10(self.mc))**2 / self.sigma**2
            )
        
        mask_high = mass >= 1.0
        if np.any(mask_high):
            m_high = mass[mask_high]
            A_high = self.A * np.exp(
                -0.5 * (np.log10(1.0) - np.log10(self.mc))**2 / self.sigma**2
            )
            imf[mask_high] = A_high * m_high**(-self.alpha)
        
        return imf if len(imf) > 1 else imf[0]
    
    def integrate(self, m_min, m_max):
        result, _ = quad(self, m_min, m_max, limit=100)
        return result


class StellarEvolution:
    """Stellar evolution"""
    
    @staticmethod
    def main_sequence_lifetime(mass):
        return 10.0 * mass**(-2.5)
    
    @staticmethod
    def turnoff_mass(age_gyr):
        return (10.0 / age_gyr)**(1.0 / 2.5)


# =============================================================================
# TNG API Loader (캐시 강화 버전)
# =============================================================================

class TNGAPILoader:
    """TNG API 로더 - 강력한 캐시 시스템"""
    
    def __init__(self, api_key, simulation='TNG50-1', snapshot=99, cache_dir='tng_cache'):
        self.api_key = api_key
        self.simulation = simulation
        self.snapshot = snapshot
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://www.tng-project.org/api"
        self.headers = {"api-key": self.api_key}
        
        # 캐시 매니저
        self.cache_manager = CacheManager(cache_dir)
        
        # 재시도 설정
        self.max_retries = 5
        self.retry_delay = 2
        
        print(f"TNG API Loader 초기화")
        print(f"  시뮬레이션: {simulation}")
        print(f"  스냅샷: {snapshot}")
        
        self._test_connection()
        
        # 캐시 정보 출력
        self.cache_manager.print_cache_info()
    
    def _test_connection(self):
        """API 연결 테스트"""
        try:
            url = f"{self.base_url}/{self.simulation}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                print(f"✓ API 연결 성공")
            elif response.status_code == 401:
                print(f"✗ API Key 오류")
                raise ValueError("Invalid API key")
            else:
                print(f"✗ API 연결 실패: {response.status_code}")
                raise ConnectionError(f"API connection failed")
        except requests.exceptions.RequestException as e:
            print(f"✗ 네트워크 오류: {e}")
            raise
    
    def _make_request_with_retry(self, url, params=None):
        """재시도 로직"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=60)
                
                if response.status_code == 200:
                    return response
                elif response.status_code in [502, 503]:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"\n  ⚠️  서버 오류. {wait_time}초 후 재시도... ({attempt+1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"서버 오류: {response.status_code}")
                elif response.status_code == 429:
                    wait_time = 60
                    print(f"\n  ⚠️  Rate limit. {wait_time}초 대기...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"API 오류: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"\n  ⚠️  Timeout. 재시도... ({attempt+1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"\n  ⚠️  네트워크 오류. 재시도...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        raise RuntimeError("최대 재시도 초과")
    
    def get_subhalo_catalog(self):
        """서브할로 카탈로그 - 캐시 우선"""
        print("\n서브할로 카탈로그 로딩...")
        
        cache_file = self.cache_dir / f"subhalos_snap{self.snapshot}.json"
        
        # 캐시 확인
        if cache_file.exists() and self.cache_manager.is_cached('subhalo_catalog'):
            print(f"  💾 캐시에서 로드 (다운로드 없음)")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"  ✓ {len(data)}개 서브할로 로드 완료")
            return data
        
        # API 다운로드
        print(f"  🌐 API에서 다운로드 중... (처음이므로 시간이 걸립니다)")
        
        url = f"{self.base_url}/{self.simulation}/snapshots/{self.snapshot}/subhalos/"
        
        all_subhalos = []
        page = 0
        
        while url:
            page += 1
            if page % 10 == 0:
                print(f"    페이지 {page}... ({len(all_subhalos)}개)")
            
            response = self._make_request_with_retry(url)
            data = response.json()
            
            all_subhalos.extend(data['results'])
            url = data['next']
            
            time.sleep(0.2)
        
        # 캐시 저장
        with open(cache_file, 'w') as f:
            json.dump(all_subhalos, f)
        
        self.cache_manager.mark_cached('subhalo_catalog')
        
        print(f"  ✓ {len(all_subhalos)}개 다운로드 완료 및 캐시 저장")
        print(f"  💡 다음 실행부터는 즉시 로드됩니다!")
        
        return all_subhalos
    
    def get_group_catalog(self):
        """그룹 카탈로그 - 캐시 우선"""
        print("\n그룹 카탈로그 로딩...")
        
        cache_file = self.cache_dir / f"groups_snap{self.snapshot}.json"
        
        # 캐시 확인
        if cache_file.exists() and self.cache_manager.is_cached('group_catalog'):
            print(f"  💾 캐시에서 로드 (다운로드 없음)")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"  ✓ {len(data)}개 그룹 로드 완료")
            return data
        
        # API 다운로드
        print(f"  🌐 API에서 다운로드 중...")
        
        url = f"{self.base_url}/{self.simulation}/snapshots/{self.snapshot}/halos/"
        
        all_groups = []
        page = 0
        
        while url:
            page += 1
            if page % 10 == 0:
                print(f"    페이지 {page}... ({len(all_groups)}개)")
            
            response = self._make_request_with_retry(url)
            data = response.json()
            
            all_groups.extend(data['results'])
            url = data['next']
            
            time.sleep(0.2)
        
        # 캐시 저장
        with open(cache_file, 'w') as f:
            json.dump(all_groups, f)
        
        self.cache_manager.mark_cached('group_catalog')
        
        print(f"  ✓ {len(all_groups)}개 다운로드 완료 및 캐시 저장")
        
        return all_groups
    
    def get_stellar_particles(self, subhalo_id):
        """별 입자 - 캐시 우선"""
        cache_file = self.cache_dir / f"stars_subhalo_{subhalo_id}_snap{self.snapshot}.npz"
        
        # 캐시 확인
        if cache_file.exists() and self.cache_manager.is_cached('galaxy', subhalo_id):
            print(f"  💾 캐시에서 로드 (다운로드 없음)")
            data = np.load(cache_file)
            stellar_data = {key: data[key] for key in data.files}
            n_stars = len(stellar_data.get('Masses', []))
            print(f"  ✓ {n_stars:,}개 별 입자 로드 완료")
            return stellar_data
        
        # API 다운로드
        print(f"  🌐 별 입자 다운로드 중...")
        
        url = f"{self.base_url}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{subhalo_id}/cutout.hdf5"
        
        params = {
            'stars': 'Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity'
        }
        
        response = self._make_request_with_retry(url, params=params)
        
        temp_file = self.cache_dir / f"temp_{subhalo_id}.hdf5"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        stellar_data = {}
        with h5py.File(temp_file, 'r') as f:
            if 'PartType4' in f:
                part4 = f['PartType4']
                
                for key in ['Coordinates', 'Velocities', 'Masses', 
                           'GFM_StellarFormationTime', 'GFM_InitialMass', 'GFM_Metallicity']:
                    if key in part4:
                        stellar_data[key] = part4[key][:]
        
        temp_file.unlink()
        
        # 캐시 저장
        np.savez_compressed(cache_file, **stellar_data)
        self.cache_manager.mark_cached('galaxy', subhalo_id)
        
        n_stars = len(stellar_data.get('Masses', []))
        print(f"  ✓ {n_stars:,}개 별 입자 다운로드 완료")
        
        return stellar_data
    
    def get_subhalo_details(self, subhalo_id):
        """서브할로 상세 정보"""
        cache_file = self.cache_dir / f"subhalo_{subhalo_id}_snap{self.snapshot}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        url = f"{self.base_url}/{self.simulation}/snapshots/{self.snapshot}/subhalos/{subhalo_id}"
        
        response = self._make_request_with_retry(url)
        data = response.json()
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data


# =============================================================================
# 나머지 클래스들은 이전과 동일
# (TNGAPIAdvancedAnalysis 등)
# =============================================================================

class TNGAPIAdvancedAnalysis:
    """TNG API + 고급 분석"""
    
    def __init__(self, api_key, simulation='TNG50-1', snapshot=99, cache_dir='tng_api_cache'):
        self.loader = TNGAPILoader(api_key, simulation, snapshot, cache_dir)
        self.h = 0.6774
        self.imf = ChabrierIMF()
        
        self.subhalo_catalog = None
        self.group_catalog = None
        self.selected_subhalos = []
        self.results = {'galaxies': [], 'summary': {}}
    
    def load_catalogs(self):
        """카탈로그 로드"""
        print("\n" + "="*70)
        print("카탈로그 로딩")
        print("="*70)
        
        self.subhalo_catalog = self.loader.get_subhalo_catalog()
        self.group_catalog = self.loader.get_group_catalog()
        
        print("\n✓ 카탈로그 로딩 완료")
    
    def select_milkyway_like_galaxies(self):
        """Milky Way-like 은하 선택"""
        print("\n" + "="*70)
        print("Milky Way-like 은하 선택")
        print("="*70)
        
        print("\n=== 1. Central Subhalo 선택 ===")
        central_ids = set()
        for group in self.group_catalog:
            first_sub = group.get('id')
            if first_sub is not None:
                central_ids.add(first_sub)
        
        print(f"Central: {len(central_ids):,}개")
        
        print("\n=== 2. 별 질량 & SFR 필터링 ===")
        candidates = []
        
        for subhalo in self.subhalo_catalog:
            sub_id = subhalo['id']
            
            if sub_id not in central_ids:
                continue
            
            mass_type = subhalo.get('mass_type', [0]*6)
            stellar_mass = mass_type[4] * 1e10 / self.h
            
            if stellar_mass < 4e10 or stellar_mass > 8e10:
                continue
            
            sfr = subhalo.get('sfr', 0)
            if sfr < 1.0 or sfr > 2.0:
                continue
            
            candidates.append({
                'id': sub_id,
                'stellar_mass': stellar_mass,
                'sfr': sfr
            })
        
        print(f"통과: {len(candidates)}개")
        
        self.selected_subhalos = [c['id'] for c in candidates]
        
        if len(candidates) > 0:
            print("\n선택된 은하 (처음 5개):")
            for i, c in enumerate(candidates[:5]):
                # 캐시 확인
                cached = "💾" if self.loader.cache_manager.is_cached('galaxy', c['id']) else "🌐"
                print(f"  {cached} {i+1}. ID {c['id']}: M* = {c['stellar_mass']:.2e} Msun, SFR = {c['sfr']:.2f} Msun/yr")
            if len(candidates) > 5:
                print(f"  ... 총 {len(candidates)}개")
            
            # 캐시된 은하 수
            cached_count = sum(1 for c in candidates if self.loader.cache_manager.is_cached('galaxy', c['id']))
            if cached_count > 0:
                print(f"\n  💡 {cached_count}개 은하는 이미 캐시에 있습니다 (다운로드 불필요)")
        
        print("\n" + "="*70)
        print(f"최종 선택: {len(self.selected_subhalos)}개")
        print("="*70)
        
        return self.selected_subhalos
    
    def analyze_galaxy(self, subhalo_id, mass_bins):
        """단일 은하 분석"""
        print(f"\n{'='*70}")
        print(f"서브할로 {subhalo_id} 분석")
        print(f"{'='*70}")
        
        details = self.loader.get_subhalo_details(subhalo_id)
        stellar_mass = details.get('mass_stars', 0) * 1e10 / self.h
        sfr = details.get('sfr', 0)
        
        print(f"  별 질량: {stellar_mass:.2e} Msun")
        print(f"  SFR: {sfr:.2f} Msun/yr")
        
        stellar_data = self.loader.get_stellar_particles(subhalo_id)
        
        if 'GFM_InitialMass' not in stellar_data:
            print("  ✗ 별 입자 데이터 없음")
            return None
        
        initial_masses = stellar_data['GFM_InitialMass'] * 1e10 / self.h
        formation_times = stellar_data['GFM_StellarFormationTime']
        
        valid_mask = formation_times > 0
        initial_masses = initial_masses[valid_mask]
        formation_times = formation_times[valid_mask]
        
        ages_gyr = 13.8 * (1.0 - formation_times)
        
        print(f"  별 입자: {len(initial_masses):,}개")
        
        counts = self._calculate_stellar_counts(initial_masses, ages_gyr, mass_bins)
        
        return {
            'subhalo_id': int(subhalo_id),
            'stellar_mass': float(stellar_mass),
            'sfr': float(sfr),
            'counts': counts
        }
    
    def _calculate_stellar_counts(self, initial_masses, ages, mass_bins):
        """별 개수 재구성"""
        print("\n  별 개수 재구성...")
        
        n_bins = len(mass_bins) - 1
        N_init_bins = np.zeros(n_bins)
        N_surv_bins = np.zeros(n_bins)
        
        for i in range(n_bins):
            m_low = mass_bins[i]
            m_high = mass_bins[i+1]
            
            mask_bin = (initial_masses >= m_low) & (initial_masses < m_high)
            
            if not np.any(mask_bin):
                continue
            
            total_mass = initial_masses[mask_bin].sum()
            
            N_imf = self.imf.integrate(m_low, m_high)
            
            def mass_weighted_imf(m):
                return m * self.imf(m)
            
            M_weighted, _ = quad(mass_weighted_imf, m_low, m_high, limit=100)
            M_avg_imf = M_weighted / N_imf if N_imf > 0 else (m_low + m_high) / 2
            
            N_init_bins[i] = total_mass / M_avg_imf
            
            ages_bin = ages[mask_bin]
            masses_bin = initial_masses[mask_bin]
            
            N_surviving = 0
            for age, mass in zip(ages_bin, masses_bin):
                if age > 0:
                    m_to = StellarEvolution.turnoff_mass(age)
                    if mass < m_to:
                        N_surviving += mass / M_avg_imf
                else:
                    N_surviving += mass / M_avg_imf
            
            N_surv_bins[i] = N_surviving
        
        N_init_total = N_init_bins.sum()
        N_surv_total = N_surv_bins.sum()
        
        survival_rate_bins = np.divide(N_surv_bins, N_init_bins,
                                       out=np.zeros_like(N_surv_bins),
                                       where=N_init_bins>0)
        
        print(f"    초기: {N_init_total:.2e}, 생존: {N_surv_total:.2e}, 생존율: {N_surv_total/N_init_total*100:.1f}%")
        
        return {
            'mass_bins': mass_bins,
            'N_init': N_init_bins,
            'N_surv': N_surv_bins,
            'survival_rate': survival_rate_bins,
            'N_init_total': N_init_total,
            'N_surv_total': N_surv_total
        }
    
    def analyze_all_galaxies(self, mass_bins=None, max_galaxies=None):
        """모든 은하 분석"""
        if mass_bins is None:
            mass_bins = np.array([0.08, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        
        galaxies_to_analyze = self.selected_subhalos[:max_galaxies] if max_galaxies else self.selected_subhalos
        
        print(f"\n총 {len(galaxies_to_analyze)}개 은하 분석")
        
        for i, subhalo_id in enumerate(galaxies_to_analyze):
            print(f"\n[{i+1}/{len(galaxies_to_analyze)}]")
            
            try:
                result = self.analyze_galaxy(subhalo_id, mass_bins)
                if result:
                    self.results['galaxies'].append(result)
            except Exception as e:
                print(f"  ✗ 분석 실패: {e}")
                continue
        
        self._compute_summary()
        
        return self.results
    
    def _compute_summary(self):
        """요약 통계"""
        if len(self.results['galaxies']) == 0:
            return
        
        n = len(self.results['galaxies'])
        
        stellar_masses = [g['stellar_mass'] for g in self.results['galaxies']]
        sfrs = [g['sfr'] for g in self.results['galaxies']]
        N_init_totals = [g['counts']['N_init_total'] for g in self.results['galaxies']]
        N_surv_totals = [g['counts']['N_surv_total'] for g in self.results['galaxies']]
        
        self.results['summary'] = {
            'n_galaxies': n,
            'stellar_mass_mean': float(np.mean(stellar_masses)),
            'stellar_mass_std': float(np.std(stellar_masses)),
            'sfr_mean': float(np.mean(sfrs)),
            'sfr_std': float(np.std(sfrs)),
            'N_init_mean': float(np.mean(N_init_totals)),
            'N_surv_mean': float(np.mean(N_surv_totals)),
            'survival_rate_mean': float(np.mean(N_surv_totals) / np.mean(N_init_totals))
        }
        
        print("\n" + "="*70)
        print("요약 통계")
        print("="*70)
        print(f"분석 은하: {n}개")
        print(f"평균 별 질량: {self.results['summary']['stellar_mass_mean']:.2e} Msun")
        print(f"평균 SFR: {self.results['summary']['sfr_mean']:.2f} Msun/yr")
        print(f"평균 생존율: {self.results['summary']['survival_rate_mean']*100:.1f}%")
    
    def save_results(self, output_dir='tng_api_results'):
        """결과 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n결과 저장: {output_dir}")
        
        json_file = output_dir / 'analysis_results.json'
        results_to_save = {
            'summary': self.results['summary'],
            'galaxies': []
        }
        
        for g in self.results['galaxies']:
            results_to_save['galaxies'].append({
                'subhalo_id': g['subhalo_id'],
                'stellar_mass': g['stellar_mass'],
                'sfr': g['sfr'],
                'counts': {
                    'mass_bins': g['counts']['mass_bins'].tolist(),
                    'N_init': g['counts']['N_init'].tolist(),
                    'N_surv': g['counts']['N_surv'].tolist(),
                    'survival_rate': g['counts']['survival_rate'].tolist(),
                    'N_init_total': g['counts']['N_init_total'],
                    'N_surv_total': g['counts']['N_surv_total']
                }
            })
        
        with open(json_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"  ✓ {json_file}")
        
        for g in self.results['galaxies']:
            csv_file = output_dir / f"galaxy_{g['subhalo_id']}_bins.csv"
            
            mass_bins = g['counts']['mass_bins']
            df = pd.DataFrame({
                'mass_bin_low': mass_bins[:-1],
                'mass_bin_high': mass_bins[1:],
                'N_init': g['counts']['N_init'],
                'N_surv': g['counts']['N_surv'],
                'survival_rate': g['counts']['survival_rate']
            })
            
            df.to_csv(csv_file, index=False)
        
        print(f"  ✓ galaxy_*_bins.csv ({len(self.results['galaxies'])}개)")
    
    def plot_results(self, output_dir='tng_api_results'):
        """플롯 생성"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n플롯 생성: {output_dir}")
        
        for g in self.results['galaxies']:
            subhalo_id = g['subhalo_id']
            mass_bins = g['counts']['mass_bins']
            survival_rate = g['counts']['survival_rate']
            
            mass_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(mass_centers, survival_rate * 100, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Stellar Mass (M$_\\odot$)', fontsize=14)
            ax.set_ylabel('Survival Rate (%)', fontsize=14)
            ax.set_title(f'Stellar Survival Rate - Subhalo {subhalo_id}', fontsize=16)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'survival_rate_subhalo_{subhalo_id}.png', dpi=300)
            plt.close()
        
        print(f"  ✓ survival_rate_subhalo_*.png ({len(self.results['galaxies'])}개)")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행"""
    print("="*80)
    print("TNG API 분석 - 캐시 최적화 버전")
    print("="*80)
    
    API_KEY = "f62123ebe9f9efb18d3ed3567e241450"
    
    simulation = 'TNG50-1'
    snapshot = 99
    cache_dir = 'tng_api_cache'
    output_dir = 'tng_api_results'
    
    mass_bins = np.array([0.08, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    
    try:
        choice = input("\n분석할 은하 수 [3/5/10/all, 기본값=3]: ").strip().lower()
        
        if choice == 'all':
            max_galaxies = None
        elif choice == '5':
            max_galaxies = 5
        elif choice == '10':
            max_galaxies = 10
        else:
            max_galaxies = 3
        
        print(f"→ {max_galaxies if max_galaxies else '모든'} 은하 분석")
        
    except:
        max_galaxies = 3
    
    try:
        analyzer = TNGAPIAdvancedAnalysis(API_KEY, simulation, snapshot, cache_dir)
        
        print("\n[1/4] 카탈로그 다운로드...")
        analyzer.load_catalogs()
        
        print("\n[2/4] 은하 선택...")
        selected = analyzer.select_milkyway_like_galaxies()
        
        if len(selected) == 0:
            print("\n⚠️  선택된 은하 없음")
            return
        
        print(f"\n[3/4] 분석...")
        analyzer.analyze_all_galaxies(mass_bins, max_galaxies=max_galaxies)
        
        if len(analyzer.results['galaxies']) == 0:
            print("\n⚠️  분석 실패")
            return
        
        print(f"\n[4/4] 저장...")
        analyzer.save_results(output_dir)
        analyzer.plot_results(output_dir)
        
        print("\n" + "="*80)
        print("✓ 완료!")
        print("="*80)
        
        # 최종 캐시 정보
        analyzer.loader.cache_manager.print_cache_info()
        
        summary = analyzer.results['summary']
        print(f"\n📊 결과: {output_dir}/")
        print(f"  • 분석 은하: {summary['n_galaxies']}개")
        print(f"  • 평균 생존율: {summary['survival_rate_mean']*100:.1f}%")
        
        print(f"\n💡 다음 실행시에는 캐시된 데이터를 사용하여 훨씬 빠릅니다!")
        
    except KeyboardInterrupt:
        print("\n\n중단됨")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
