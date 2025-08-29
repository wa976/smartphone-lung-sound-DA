import math
import numpy as np
from torch.utils.data import Sampler


class BalancedDomainSampler(Sampler):
    def __init__(self, dataset, batch_size, domain_ratios=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_labels = self._get_domain_labels()
        self.domain_indices = self._get_domain_indices()
        
        # 도메인 비율 설정 (기본값은 균등 비율)
        if domain_ratios is None:
            self.domain_ratios = {domain: 1 for domain in set(self.domain_labels)}
        else:
            self.domain_ratios = domain_ratios
        
        # 배치 내 각 도메인별 샘플 수 계산
        self.samples_per_domain = self._get_samples_per_domain()
        
        # 전체 반복 횟수 계산
        self.total_batches = self._calculate_total_batches()
    
    def _get_domain_labels(self):
        """데이터셋에서 도메인 라벨 추출"""
        domain_labels = []
        for i in range(len(self.dataset)):
            _, labels = self.dataset[i]
            if isinstance(labels, tuple):
                domain_labels.append(labels[1].item())  # 도메인 라벨은 두 번째 요소
            else:
                domain_labels.append(labels.item())
        return domain_labels
    
    def _get_domain_indices(self):
        """도메인별 인덱스 목록 생성"""
        domain_indices = {}
        for idx, domain in enumerate(self.domain_labels):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)
        return domain_indices
    
    def _get_samples_per_domain(self):
        """배치 내 각 도메인별 샘플 수 계산"""
        total_ratio = sum(self.domain_ratios.values())
        samples_per_domain = {}
        
        for domain, ratio in self.domain_ratios.items():
            # 비율에 따라 샘플 수 계산 (최소 1개)
            samples_per_domain[domain] = max(1, int(self.batch_size * (ratio / total_ratio)))
        
        # 배치 크기에 맞게 조정
        total_samples = sum(samples_per_domain.values())
        if total_samples != self.batch_size:
            # 가장 많은 샘플을 가진 도메인에서 조정
            max_domain = max(samples_per_domain, key=samples_per_domain.get)
            samples_per_domain[max_domain] += (self.batch_size - total_samples)
        
        return samples_per_domain
    
    def _calculate_total_batches(self):
        """전체 배치 수 계산 (가장 적은 샘플을 가진 도메인 기준)"""
        min_batches = float('inf')
        for domain, indices in self.domain_indices.items():
            if domain in self.samples_per_domain and self.samples_per_domain[domain] > 0:
                domain_batches = len(indices) // self.samples_per_domain[domain]
                min_batches = min(min_batches, domain_batches)
        
        return max(1, min_batches)  # 최소 1개 배치 보장
    
    def __iter__(self):
        """배치 생성 반복자"""
        # 각 도메인별 인덱스를 섞음
        domain_iterators = {}
        for domain, indices in self.domain_indices.items():
            domain_iterators[domain] = iter(np.random.permutation(indices))
        
        all_indices = []
        
        # 각 배치에 대해
        for _ in range(self.total_batches):
            batch_indices = []
            
            # 각 도메인에서 지정된 수만큼 샘플 추출
            for domain, num_samples in self.samples_per_domain.items():
                if domain in domain_iterators:
                    iterator = domain_iterators[domain]
                    domain_batch = []
                    
                    # 필요한 샘플 수만큼 추출
                    for _ in range(num_samples):
                        try:
                            idx = next(iterator)
                            domain_batch.append(idx)
                        except StopIteration:
                            # 도메인의 모든 샘플을 사용했으면 다시 섞어서 사용
                            domain_iterators[domain] = iter(np.random.permutation(self.domain_indices[domain]))
                            idx = next(domain_iterators[domain])
                            domain_batch.append(idx)
                    
                    batch_indices.extend(domain_batch)
            
            # 배치 내 샘플 순서 섞기
            np.random.shuffle(batch_indices)
            all_indices.extend(batch_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        """전체 샘플 수 반환"""
        return self.total_batches * self.batch_size


class BalancedDomainClassSampler(Sampler):
    """도메인과 클래스 비율을 모두 고려하는 샘플러 (기존 코드)"""
    # 기존 코드 유지
    pass