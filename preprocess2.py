import os
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import random
import re
import math
import time
from datetime import datetime

# MediaPipe 초기화 (필요한 경우 외부에서 초기화하고 전달)
# 여기서는 편의상 포함
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- 유틸리티 및 전처리 함수 ---

def get_rotation_matrix_3d(angle_degrees):
    """지정된 각도로 Z축을 중심으로 회전하는 3D 회전 행렬 생성"""
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

def augment_scale(landmarks, scale_factor):
    """랜드마크 좌표에 특정 스케일링 적용 (중앙 기준)"""
    if landmarks.size == 0:
        return landmarks
    aug_lms = landmarks.copy()
    center = np.mean(landmarks[:, :3], axis=0, keepdims=True)
    aug_lms[:, :3] = center + (aug_lms[:, :3] - center) * scale_factor
    return aug_lms

def augment_jitter(landmarks, sigma=0.015):
    """랜드마크 좌표에 노이즈 추가"""
    if landmarks.size == 0:
        return landmarks
    noise = np.random.normal(0, sigma, landmarks[:, :3].shape)
    aug_lms = landmarks.copy()
    aug_lms[:, :3] = aug_lms[:, :3] + noise
    return aug_lms

def augment_rotate(landmarks, angle_degrees):
    """랜드마크 좌표를 지정된 각도로 3D 회전"""
    if landmarks.size == 0:
        return landmarks
    rotation_matrix = get_rotation_matrix_3d(angle_degrees)
    center = np.mean(landmarks[:, :3], axis=0, keepdims=True)
    rotated_lms = landmarks.copy()
    rotated_lms[:, :3] = center + np.dot(landmarks[:, :3] - center, rotation_matrix.T)
    return rotated_lms

def augment_time_warp(sequence, warp_factor=1.0):
    """시퀀스의 길이를 warp_factor에 따라 조정 (보간 사용)"""
    if warp_factor == 1.0 or len(sequence) == 0:
        return sequence
    orig_len = len(sequence)
    new_len = int(orig_len * warp_factor)
    if new_len == 0:
        return sequence[:1]
    indices = np.linspace(0, orig_len - 1, new_len)
    
    warped_seq = np.array([sequence[int(i)] + (sequence[min(int(i)+1, orig_len-1)] - sequence[int(i)]) * (i - int(i)) for i in indices])
    return warped_seq

def augment_occlusion_mask(landmarks, occlusion_prob=0.1):
    """
    랜드마크 중 무작위로 occlusion_prob 확률로 마스크를 생성하고 랜드마크를 0으로 설정
    """
    if landmarks.size == 0:
        return landmarks, np.zeros(landmarks.shape[:1], dtype=np.bool_)
    
    aug_lms = landmarks.copy()
    mask = np.random.rand(aug_lms.shape[0]) > occlusion_prob
    
    # 마스크가 False인 랜드마크의 좌표를 0으로 설정
    aug_lms[~mask, :3] = 0.0
    
    return aug_lms, mask

def augment_mirror_hands(pose_seq, left_hand_seq, right_hand_seq):
    """
    손 랜드마크를 좌우 반전시키고, 오른손과 왼손 랜드마크를 교체
    """
    mirrored_pose = pose_seq.copy()
    mirrored_pose[:, :, 0] *= -1 # x좌표 반전
    
    mirrored_left_hand = right_hand_seq.copy()
    mirrored_right_hand = left_hand_seq.copy()
    mirrored_left_hand[:, :, 0] *= -1
    mirrored_right_hand[:, :, 0] *= -1
    
    # 오른손/왼손 랜드마크 교체
    temp_hand = mirrored_left_hand
    mirrored_left_hand = mirrored_right_hand
    mirrored_right_hand = temp_hand
    
    # 포즈 랜드마크 중 팔/어깨도 교체
    mirrored_pose[:, [1, 2]] = mirrored_pose[:, [2, 1]]
    mirrored_pose[:, [3, 4]] = mirrored_pose[:, [4, 3]]
    mirrored_pose[:, [5, 6]] = mirrored_pose[:, [6, 5]]
    
    return mirrored_pose, mirrored_left_hand, mirrored_right_hand

def normalize_landmarks(pose_lms, left_hand_lms, right_hand_lms):
    """
    모든 랜드마크를 어깨 중앙을 기준으로 정규화
    """
    if pose_lms.shape[0] < 3 or (np.sum(pose_lms[1, :3]) == 0 and np.sum(pose_lms[2, :3]) == 0):
        return pose_lms, left_hand_lms, right_hand_lms

    left_shoulder = pose_lms[1, :3]
    right_shoulder = pose_lms[2, :3]
    
    if np.sum(left_shoulder) == 0:
        center_shoulder = right_shoulder
    elif np.sum(right_shoulder) == 0:
        center_shoulder = left_shoulder
    else:
        center_shoulder = (left_shoulder + right_shoulder) / 2.0

    norm_pose_lms = pose_lms.copy()
    norm_left_hand_lms = left_hand_lms.copy()
    norm_right_hand_lms = right_hand_lms.copy()
    
    norm_pose_lms[:, :3] = norm_pose_lms[:, :3] - center_shoulder
    norm_left_hand_lms[:, :3] = norm_left_hand_lms[:, :3] - center_shoulder
    norm_right_hand_lms[:, :3] = norm_right_hand_lms[:, :3] - center_shoulder

    return norm_pose_lms, norm_left_hand_lms, norm_right_hand_lms

def extract_landmarks(image):
    """Pose와 Hands에서 랜드마크 추출 (PIL 이미지 입력)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    pose_results = pose.process(image_np)
    hands_results = hands.process(image_np)

    # 랜드마크와 가시성(visibility)을 함께 저장
    pose_landmarks = np.zeros((7, 4)) 
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        selected_indices = [0, 11, 12, 13, 14, 15, 16] # 코, 양 어깨, 양 팔꿈치, 양 손목
        for i, idx in enumerate(selected_indices):
            pose_landmarks[i] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z, landmarks[idx].visibility]

    left_hand_lms = np.zeros((21, 3))
    right_hand_lms = np.zeros((21, 3))
    
    if hands_results.multi_hand_landmarks:
        for hand_idx, hand_lm in enumerate(hands_results.multi_hand_landmarks):
            handedness_label = hands_results.multi_handedness[hand_idx].classification[0].label
            
            hand_lms_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
            
            if handedness_label == 'Right':
                right_hand_lms = hand_lms_coords
            elif handedness_label == 'Left':
                left_hand_lms = hand_lms_coords
    
    return pose_landmarks, left_hand_lms, right_hand_lms

def pad_and_trim_sequence(sequence, max_length):
    """시퀀스를 max_length로 패딩 또는 자르기"""
    if max_length is None:
        return sequence
    
    current_length = len(sequence)
    if current_length > max_length:
        return sequence[:max_length]
    
    if current_length == max_length:
        return sequence

    padding = np.zeros((max_length - current_length, *sequence.shape[1:]))
    padded_sequence = np.vstack([sequence, padding])
    
    return padded_sequence

def calculate_angle(p1, p2, p3):
    """3D 랜드마크 3개로 관절 각도 계산 (라디안)"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return angle

def calculate_kinematic_features(pose_seq, left_hand_seq, right_hand_seq):
    """
    랜드마크 시퀀스로부터 관절 각도, 랜드마크 변위, 거리, 손가락 굽힘, 손바닥 방향 피처를 계산
    """
    features_seq = []
    
    # 손가락 굽힘 각도 계산을 위한 인덱스
    finger_lms_indices = [
        # 엄지
        (0, 1, 2), (1, 2, 3), (2, 3, 4), 
        # 검지
        (0, 5, 6), (5, 6, 7), (6, 7, 8), 
        # 중지
        (0, 9, 10), (9, 10, 11), (10, 11, 12), 
        # 약지
        (0, 13, 14), (13, 14, 15), (14, 15, 16), 
        # 새끼
        (0, 17, 18), (17, 18, 19), (18, 19, 20) 
    ]
    
    for i in range(len(pose_seq)):
        frame_features = []
        
        # 1. 랜드마크 변위/속도
        if i > 0:
            pose_disp = pose_seq[i, :, :3] - pose_seq[i-1, :, :3]
            left_hand_disp = left_hand_seq[i] - left_hand_seq[i-1]
            right_hand_disp = right_hand_seq[i] - right_hand_seq[i-1]
            frame_features.extend(pose_disp.flatten())
            frame_features.extend(left_hand_disp.flatten())
            frame_features.extend(right_hand_disp.flatten())
        else:
            frame_features.extend(np.zeros_like(pose_seq[i, :, :3]).flatten())
            frame_features.extend(np.zeros_like(left_hand_seq[i]).flatten())
            frame_features.extend(np.zeros_like(right_hand_seq[i]).flatten())

        # 2. 관절 각도
        if np.sum(pose_seq[i, 2, :3]) != 0 and np.sum(pose_seq[i, 4, :3]) != 0 and np.sum(pose_seq[i, 6, :3]) != 0:
            angle_right_elbow = calculate_angle(pose_seq[i, 2, :3], pose_seq[i, 4, :3], pose_seq[i, 6, :3])
            frame_features.append(angle_right_elbow)
        else:
            frame_features.append(0.0)

        if np.sum(pose_seq[i, 1, :3]) != 0 and np.sum(pose_seq[i, 3, :3]) != 0 and np.sum(pose_seq[i, 5, :3]) != 0:
            angle_left_elbow = calculate_angle(pose_seq[i, 1, :3], pose_seq[i, 3, :3], pose_seq[i, 5, :3])
            frame_features.append(angle_left_elbow)
        else:
            frame_features.append(0.0)

        # 3. 손가락 굽힘 각도 (각 손 15개, 총 30개)
        def _calculate_finger_angles(hand_lms):
            if np.sum(hand_lms) == 0: return [0.0] * 15
            angles = []
            for p1_idx, p2_idx, p3_idx in finger_lms_indices:
                angles.append(calculate_angle(hand_lms[p1_idx], hand_lms[p2_idx], hand_lms[p3_idx]))
            return angles
            
        frame_features.extend(_calculate_finger_angles(left_hand_seq[i]))
        frame_features.extend(_calculate_finger_angles(right_hand_seq[i]))

        # 4. 손바닥 방향 벡터 (각 손 3개, 총 6개)
        def _calculate_palm_vector(hand_lms):
            if np.sum(hand_lms) == 0: return [0.0, 0.0, 0.0]
            wrist = hand_lms[0]
            palm_middle = (hand_lms[9] + hand_lms[13]) / 2
            vector = palm_middle - wrist
            norm = np.linalg.norm(vector)
            return vector / norm if norm != 0 else np.zeros(3)

        frame_features.extend(_calculate_palm_vector(left_hand_seq[i]))
        frame_features.extend(_calculate_palm_vector(right_hand_seq[i]))

        # 5. 손과 몸통 랜드마크 간의 상대 거리 (2개)
        if np.sum(pose_seq[i, 0, :3]) != 0 and np.sum(left_hand_seq[i, 0]) != 0:
            dist_left_hand_to_nose = np.linalg.norm(left_hand_seq[i, 0] - pose_seq[i, 0, :3])
            frame_features.append(dist_left_hand_to_nose)
        else:
            frame_features.append(0.0)
            
        if np.sum(pose_seq[i, 0, :3]) != 0 and np.sum(right_hand_seq[i, 0]) != 0:
            dist_right_hand_to_nose = np.linalg.norm(right_hand_seq[i, 0] - pose_seq[i, 0, :3])
            frame_features.append(dist_right_hand_to_nose)
        else:
            frame_features.append(0.0)
        
        # 6. 랜드마크 간 거리 (1개)
        if np.sum(pose_seq[i, 5, :3]) != 0 and np.sum(pose_seq[i, 6, :3]) != 0:
            dist_wrists = np.linalg.norm(pose_seq[i, 5, :3] - pose_seq[i, 6, :3])
            frame_features.append(dist_wrists)
        else:
            frame_features.append(0.0)
        
        features_seq.append(np.array(frame_features, dtype=np.float32))
        
    return np.array(features_seq)

def augment_and_save_sequence(pose_seq, left_hand_seq, right_hand_seq, output_dir, class_name, video_name, data_info, labels, dataset_type, MAX_SEQ_LENGTH, NUM_AUGMENTATIONS_PER_VIDEO):
    """시퀀스 증강 및 npz 파일로 저장, labels.csv에 추가"""
    
    sequences_to_save = {}
    
    sequences_to_save['original'] = {
        'pose': pose_seq.copy(),
        'left_hand': left_hand_seq.copy(),
        'right_hand': right_hand_seq.copy(),
        'pose_mask': np.ones(pose_seq.shape[:2], dtype=np.bool_),
        'left_hand_mask': np.ones(left_hand_seq.shape[:2], dtype=np.bool_),
        'right_hand_mask': np.ones(right_hand_seq.shape[:2], dtype=np.bool_),
    }
    
    # 증강 설정 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    augmentation_configs = [
        ('scale', 0.8), ('scale', 1.2),
        ('jitter', 0.015), ('jitter', 0.025),
        ('rotate', 5), ('rotate', -5),
        ('time_warp', 0.8), ('time_warp', 1.2),
        ('occlusion', 0.1), ('occlusion', 0.2),
        ('mirror', None),
        ('time_reverse', None)
    ]
    
    for i in range(NUM_AUGMENTATIONS_PER_VIDEO - 1):
        aug_pose_seq = pose_seq.copy()
        aug_left_hand_seq = left_hand_seq.copy()
        aug_right_hand_seq = right_hand_seq.copy()
        
        # 마스크 초기화: 원본 시퀀스 길이를 사용하여 초기화
        pose_mask_seq = np.ones(aug_pose_seq.shape[:2], dtype=np.bool_)
        left_hand_mask_seq = np.ones(aug_left_hand_seq.shape[:2], dtype=np.bool_)
        right_hand_mask_seq = np.ones(aug_right_hand_seq.shape[:2], dtype=np.bool_)
        
        num_aug = random.randint(1, 2)
        selected_augs = random.sample(augmentation_configs, num_aug)
        
        aug_key_name = []
        for aug_type_base, param in selected_augs:
            if aug_type_base == 'scale':
                aug_pose_seq = np.array([augment_scale(frame, param) for frame in aug_pose_seq])
                aug_left_hand_seq = np.array([augment_scale(frame, param) for frame in aug_left_hand_seq])
                aug_right_hand_seq = np.array([augment_scale(frame, param) for frame in aug_right_hand_seq])
            elif aug_type_base == 'jitter':
                aug_pose_seq = np.array([augment_jitter(frame, sigma=param) for frame in aug_pose_seq])
                aug_left_hand_seq = np.array([augment_jitter(frame, sigma=param) for frame in aug_left_hand_seq])
                aug_right_hand_seq = np.array([augment_jitter(frame, sigma=param) for frame in aug_right_hand_seq])
            elif aug_type_base == 'rotate':
                aug_pose_seq = np.array([augment_rotate(frame, param) for frame in aug_pose_seq])
                aug_left_hand_seq = np.array([augment_rotate(frame, param) for frame in aug_left_hand_seq])
                aug_right_hand_seq = np.array([augment_rotate(frame, param) for frame in aug_right_hand_seq])
            elif aug_type_base == 'time_warp':
                aug_pose_seq = augment_time_warp(aug_pose_seq, param)
                aug_left_hand_seq = augment_time_warp(aug_left_hand_seq, param)
                aug_right_hand_seq = augment_time_warp(aug_right_hand_seq, param)
                
                # time_warp 적용 후 마스크 재초기화
                pose_mask_seq = np.ones(aug_pose_seq.shape[:2], dtype=np.bool_)
                left_hand_mask_seq = np.ones(aug_left_hand_seq.shape[:2], dtype=np.bool_)
                right_hand_mask_seq = np.ones(aug_right_hand_seq.shape[:2], dtype=np.bool_)
            elif aug_type_base == 'occlusion':
                # 마스크 증강 적용
                temp_pose_seq, temp_pose_mask_seq = [], []
                temp_left_hand_seq, temp_left_hand_mask_seq = [], []
                temp_right_hand_seq, temp_right_hand_mask_seq = [], []
                
                for t in range(len(aug_pose_seq)):
                    masked_pose, pose_mask = augment_occlusion_mask(aug_pose_seq[t], occlusion_prob=param)
                    masked_left_hand, left_hand_mask = augment_occlusion_mask(aug_left_hand_seq[t], occlusion_prob=param)
                    masked_right_hand, right_hand_mask = augment_occlusion_mask(aug_right_hand_seq[t], occlusion_prob=param)
                    
                    temp_pose_seq.append(masked_pose)
                    temp_pose_mask_seq.append(pose_mask)
                    temp_left_hand_seq.append(masked_left_hand)
                    temp_left_hand_mask_seq.append(left_hand_mask)
                    temp_right_hand_seq.append(masked_right_hand)
                    temp_right_hand_mask_seq.append(right_hand_mask)
                
                aug_pose_seq = np.array(temp_pose_seq)
                aug_left_hand_seq = np.array(temp_left_hand_seq)
                aug_right_hand_seq = np.array(temp_right_hand_seq)
                pose_mask_seq = np.array(temp_pose_mask_seq)
                left_hand_mask_seq = np.array(temp_left_hand_mask_seq)
                right_hand_mask_seq = np.array(temp_right_hand_mask_seq)
            elif aug_type_base == 'mirror':
                aug_pose_seq, aug_left_hand_seq, aug_right_hand_seq = augment_mirror_hands(aug_pose_seq, aug_left_hand_seq, aug_right_hand_seq)
            elif aug_type_base == 'time_reverse':
                aug_pose_seq = np.flip(aug_pose_seq, axis=0)
                aug_left_hand_seq = np.flip(aug_left_hand_seq, axis=0)
                aug_right_hand_seq = np.flip(aug_right_hand_seq, axis=0)
            
            aug_key_name.append(f'{aug_type_base}_{param if param is not None else "None"}')

        aug_key = f'aug_{i+1}_{"&".join(aug_key_name)}'
        sequences_to_save[aug_key] = {
            'pose': aug_pose_seq,
            'left_hand': aug_left_hand_seq,
            'right_hand': aug_right_hand_seq,
            'pose_mask': pose_mask_seq,
            'left_hand_mask': left_hand_mask_seq,
            'right_hand_mask': right_hand_mask_seq,
        }

    for aug_key, seqs in sequences_to_save.items():
        # 마스킹된 데이터로 특징 계산
        additional_features = calculate_kinematic_features(seqs['pose'], seqs['left_hand'], seqs['right_hand'])
        
        padded_pose_seq = pad_and_trim_sequence(seqs['pose'], MAX_SEQ_LENGTH)
        padded_left_hand_seq = pad_and_trim_sequence(seqs['left_hand'], MAX_SEQ_LENGTH)
        padded_right_hand_seq = pad_and_trim_sequence(seqs['right_hand'], MAX_SEQ_LENGTH)
        padded_features_seq = pad_and_trim_sequence(additional_features, MAX_SEQ_LENGTH)
        
        # 마스크도 패딩
        padded_pose_mask = pad_and_trim_sequence(seqs['pose_mask'], MAX_SEQ_LENGTH)
        padded_left_hand_mask = pad_and_trim_sequence(seqs['left_hand_mask'], MAX_SEQ_LENGTH)
        padded_right_hand_mask = pad_and_trim_sequence(seqs['right_hand_mask'], MAX_SEQ_LENGTH)

        # JSON 대신 npz 파일로 저장
        output_sub_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_sub_dir, exist_ok=True)
        filename_base = f"{video_name}-{aug_key}"
        output_file_npz = os.path.join(output_sub_dir, f"{filename_base}.npz")
        
        try:
            np.savez_compressed(output_file_npz,
                                 pose=padded_pose_seq,
                                 left_hand=padded_left_hand_seq,
                                 right_hand=padded_right_hand_seq,
                                 features=padded_features_seq,
                                 pose_mask=padded_pose_mask,
                                 left_hand_mask=padded_left_hand_mask,
                                 right_hand_mask=padded_right_hand_mask)
            
            data_info.append({
                'filename': os.path.join(class_name, f"{filename_base}.npz"),
                'labels': labels[0],
                'type': dataset_type,
                'length': len(seqs['pose']),
                'aug_type': aug_key
            })
        except Exception as e:
            print(f"NPZ 저장 실패: {output_file_npz} (에러: {str(e)})")

def process_sequence(video_dir, output_dir, class_name, video_name, data_info, MAX_SEQ_LENGTH, NUM_AUGMENTATIONS_PER_VIDEO):
    """영상 디렉토리의 프레임 시퀀스 처리"""
    frame_files = sorted([f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    if not frame_files:
        print(f"영상 디렉토리 {video_dir}에 .jpg 파일이 없습니다. 건너뜁니다.")
        return
    
    all_pose_lms = []
    all_left_hand_lms = []
    all_right_hand_lms = []
    
    for frame_file in frame_files:
        img_path = os.path.join(video_dir, frame_file)
        try:
            with Image.open(img_path) as img:
                pose_lms, left_hand_lms, right_hand_lms = extract_landmarks(img)
                all_pose_lms.append(pose_lms)
                all_left_hand_lms.append(left_hand_lms)
                all_right_hand_lms.append(right_hand_lms)
        except Exception as e:
            print(f"이미지 로드 또는 랜드마크 추출 실패: {img_path} (에러: {str(e)})")
            continue

    if not all_pose_lms:
        print(f"시퀀스 생성 실패: {video_dir} (유효한 랜드마크 없음). 건너뜁니다.")
        return
    
    all_pose_lms = np.array(all_pose_lms)
    all_left_hand_lms = np.array(all_left_hand_lms)
    all_right_hand_lms = np.array(all_right_hand_lms)
    
    normalized_pose_seq = []
    normalized_left_hand_seq = []
    normalized_right_hand_seq = []
    
    for frame_idx in range(len(all_pose_lms)):
        norm_pose, norm_left, norm_right = normalize_landmarks(
            all_pose_lms[frame_idx], 
            all_left_hand_lms[frame_idx], 
            all_right_hand_lms[frame_idx]
        )
        normalized_pose_seq.append(norm_pose)
        normalized_left_hand_seq.append(norm_left)
        normalized_right_hand_seq.append(norm_right)
    
    normalized_pose_seq = np.array(normalized_pose_seq)
    normalized_left_hand_seq = np.array(normalized_left_hand_seq)
    normalized_right_hand_seq = np.array(normalized_right_hand_seq)

    labels = [class_name]
    rand = random.random()
    if rand < 0.7:  # TRAIN_RATIO
        dataset_type = 'train'
    elif rand < 0.7 + 0.2: # TRAIN_RATIO + VAL_RATIO
        dataset_type = 'val'
    else:
        dataset_type = 'test'

    augment_and_save_sequence(normalized_pose_seq, normalized_left_hand_seq, normalized_right_hand_seq, output_dir, class_name, video_name, data_info, labels, dataset_type, MAX_SEQ_LENGTH, NUM_AUGMENTATIONS_PER_VIDEO)

def create_augmented_dataset(root_dir, output_dir, MAX_SEQ_LENGTH, NUM_AUGMENTATIONS_PER_VIDEO):
    """데이터셋 생성 및 증강"""
    os.makedirs(output_dir, exist_ok=True)
    data_info = []
    
    print(f"데이터셋 생성 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 변경된 부분: W_001~W_040 디렉토리를 탐색
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'W_\d{3}', d)])
    
    if not class_names:
        print(f"루트 디렉토리 {root_dir}에서 유효한 클래스 디렉토리 (W_001~W_040 형식)를 찾을 수 없습니다.")
        return

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        print(f"클래스 디렉토리 탐색: {class_dir}")
        
        # 각 클래스 디렉토리 내의 영상 디렉토리를 탐색
        video_dirs = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])
        
        if not video_dirs:
            print(f"클래스 {class_name} 디렉토리 {class_dir}에서 유효한 영상 디렉토리를 찾을 수 없습니다.")
            continue
        
        print(f"클래스 {class_name}에서 발견된 영상 디렉토리: {len(video_dirs)}개 (예시: {', '.join(video_dirs[:5]) + '...' if len(video_dirs) > 5 else ', '.join(video_dirs)})")
        
        for video_name in video_dirs:
            video_full_path = os.path.join(class_dir, video_name)
            print(f"처리 중인 영상: {video_full_path}")
            process_sequence(video_full_path, output_dir, class_name, video_name, data_info, MAX_SEQ_LENGTH, NUM_AUGMENTATIONS_PER_VIDEO)

    if data_info:
        df = pd.DataFrame(data_info)
        csv_path = os.path.join(output_dir, 'labels.csv')
        df.to_csv(csv_path, index=False)
        print(f"데이터셋 생성 완료: {len(data_info)}개의 시퀀스, labels.csv 저장됨 ({csv_path})")
    else:
        print("데이터셋 생성 실패: 유효한 시퀀스가 없습니다.")