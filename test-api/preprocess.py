import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(p1, p2, p3):
    """세 점 사이의 각도를 계산합니다."""
    # 2D 또는 3D 벡터 모두 처리 가능하도록 수정
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return angle

class FeatureExtractor:
    """MediaPipe를 사용하여 프레임에서 512차원 특징 벡터를 추출하는 클래스."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

    def _extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        
        pose_landmarks = np.zeros((9, 3))
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            selected_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24] # nose, shoulders, elbows, wrists, hips
            for i, idx in enumerate(selected_indices):
                lm = landmarks[idx]
                pose_landmarks[i] = [lm.x, lm.y, lm.z]

        left_hand_lms = np.zeros((21, 3))
        right_hand_lms = np.zeros((21, 3))
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_lm in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[hand_idx].classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
                if handedness == 'Right':
                    right_hand_lms = coords
                elif handedness == 'Left':
                    left_hand_lms = coords
        return pose_landmarks, left_hand_lms, right_hand_lms, hands_results

    def _normalize_landmarks(self, pose_lms, left_hand_lms, right_hand_lms):
        left_shoulder, right_shoulder = pose_lms[1], pose_lms[2]
        center_shoulder = right_shoulder if np.sum(left_shoulder) == 0 else \
                          left_shoulder if np.sum(right_shoulder) == 0 else \
                          (left_shoulder + right_shoulder) / 2.0
        
        norm_pose = pose_lms - center_shoulder
        norm_left = left_hand_lms - center_shoulder
        norm_right = right_hand_lms - center_shoulder
        return norm_pose, norm_left, norm_right
    
    def _calculate_finger_angles(self, hand_lms):
        if np.sum(hand_lms) == 0: return [0.0] * 15
        angles = []
        indices = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
        for p1, p2, p3 in indices: 
            angles.append(calculate_angle(hand_lms[p1], hand_lms[p2], hand_lms[p3]))
        return angles

    # --- 추가된 헬퍼 함수 1: 손바닥 방향 벡터 ---
    def _calculate_palm_vector(self, hand_lms):
        if np.sum(hand_lms) == 0:
            return np.zeros(3)
        # 손목(0), 검지뿌리(5), 새끼뿌리(17)
        p0, p5, p17 = hand_lms[0], hand_lms[5], hand_lms[17]
        v1 = p5 - p0
        v2 = p17 - p0
        palm_normal = np.cross(v1, v2)
        norm = np.linalg.norm(palm_normal)
        return palm_normal / norm if norm != 0 else np.zeros(3)

    # --- 추가된 헬퍼 함수 2: 손 모양 서술자 ---
    def _hand_shape_descriptor(self, hand_lms):
        if np.sum(hand_lms) == 0:
            return np.zeros(5)
        wrist = hand_lms[0]
        finger_tips = hand_lms[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(finger_tips - wrist, axis=1)
        return distances.tolist()

    def _calculate_kinematic_features_for_frame(self, pose_frame, left_frame, right_frame, prev_data):
        frame_features = []
        
        # 1. 속도 (153차원)
        current_velocity = np.zeros(9*3 + 21*3 + 21*3)
        if prev_data:
            pose_disp = pose_frame - prev_data['pose']
            left_disp = left_frame - prev_data['left']
            right_disp = right_frame - prev_data['right']
            current_velocity = np.concatenate([d.flatten() for d in [pose_disp, left_disp, right_disp]])
        frame_features.extend(current_velocity)

        # 2. 가속도 (153차원)
        acceleration = np.zeros_like(current_velocity)
        if prev_data and 'velocity' in prev_data:
            acceleration = current_velocity - prev_data['velocity']
        frame_features.extend(acceleration)

        # 3. 관절 각도 (3차원) --- 수정된 부분 ---
        angles = []
        # 팔꿈치 각도
        indices = [(2, 4, 6), (1, 3, 5)] # 오른쪽, 왼쪽 팔꿈치
        for p1, p2, p3 in indices:
            if np.any(pose_frame[p1]) and np.any(pose_frame[p2]) and np.any(pose_frame[p3]):
                angles.append(calculate_angle(pose_frame[p1], pose_frame[p2], pose_frame[p3]))
            else: 
                angles.append(0.0)
        # 어깨 기울기 각도
        if np.any(pose_frame[1]) and np.any(pose_frame[2]):
            shoulder_center = np.mean(pose_frame[[1,2]], axis=0)
            angles.append(calculate_angle(pose_frame[1][:2], shoulder_center[:2], pose_frame[2][:2]))
        else:
            angles.append(0.0)
        frame_features.extend(angles)

        # 4. 손가락 각도 (30차원)
        frame_features.extend(self._calculate_finger_angles(left_frame))
        frame_features.extend(self._calculate_finger_angles(right_frame))

        # --- 추가된 부분 1: 손바닥 방향 벡터 (6차원) ---
        frame_features.extend(self._calculate_palm_vector(left_frame))
        frame_features.extend(self._calculate_palm_vector(right_frame))
        
        # 5. 랜드마크 간 거리 (4차원) --- 수정된 부분 ---
        distances = []
        nose = pose_frame[0]
        # 손목 좌표는 pose_frame에서 가져오는 것이 더 안정적
        l_wrist_pose, r_wrist_pose = pose_frame[5], pose_frame[6] 
        l_hip, r_hip = pose_frame[7], pose_frame[8]

        distances.append(np.linalg.norm(l_wrist_pose - nose) if np.any(nose) and np.any(l_wrist_pose) else 0.0)
        distances.append(np.linalg.norm(r_wrist_pose - nose) if np.any(nose) and np.any(r_wrist_pose) else 0.0)
        distances.append(np.linalg.norm(l_wrist_pose - r_wrist_pose) if np.any(l_wrist_pose) and np.any(r_wrist_pose) else 0.0)
        distances.append(np.linalg.norm(l_hip - r_hip) if np.any(l_hip) and np.any(r_hip) else 0.0)
        frame_features.extend(distances)
        
        # --- 추가된 부분 2: 손 모양 서술자 (10차원) ---
        frame_features.extend(self._hand_shape_descriptor(left_frame))
        frame_features.extend(self._hand_shape_descriptor(right_frame))

        return np.array(frame_features, dtype=np.float32), current_velocity

    def get_combined_features(self, image, prev_data):
        """단일 프레임에서 최종 512차원 특징 벡터를 추출합니다."""
        pose_lms, left_lms, right_lms, hands_results = self._extract_landmarks(image)
        norm_pose, norm_left, norm_right = self._normalize_landmarks(pose_lms, left_lms, right_lms)
        
        kinematic_features, current_velocity = self._calculate_kinematic_features_for_frame(
            norm_pose, norm_left, norm_right, prev_data
        )

        # 최종적으로 모든 특징을 하나로 합침
        combined_vector = np.concatenate([
            norm_pose.flatten(),        # 27
            norm_left.flatten(),        # 63
            norm_right.flatten(),       # 63
            kinematic_features          # 359
        ]).astype(np.float32)

        current_data = {
            'pose': norm_pose, 'left': norm_left, 'right': norm_right, 'velocity': current_velocity
        }
        
        return combined_vector, current_data, hands_results