import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_angle_3d(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return angle

class FeatureExtractor:
    def __init__(self, scaler_path=None):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def _extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        
        pose_landmarks = np.zeros((9, 3))
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            selected_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
            for i, idx in enumerate(selected_indices):
                lm = landmarks[idx]
                pose_landmarks[i] = [lm.x, lm.y, lm.z if lm.z is not None else 0]

        left_hand_lms = np.zeros((21, 3))
        right_hand_lms = np.zeros((21, 3))
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_lm in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[hand_idx].classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z if lm.z is not None else 0] for lm in hand_lm.landmark])
                if handedness == 'Right':
                    right_hand_lms = coords
                elif handedness == 'Left':
                    left_hand_lms = coords
        return pose_landmarks, left_hand_lms, right_hand_lms, hands_results

    def _normalize_landmarks(self, pose_lms, left_hand_lms, right_hand_lms):
        if pose_lms.shape[0] < 3 or (np.sum(pose_lms[1]) == 0 and np.sum(pose_lms[2]) == 0):
            return pose_lms, left_hand_lms, right_hand_lms
        left_shoulder, right_shoulder = pose_lms[1], pose_lms[2]
        center_shoulder = (left_shoulder + right_shoulder) / 2.0
        return pose_lms - center_shoulder, left_hand_lms - center_shoulder, right_hand_lms - center_shoulder

    def _calculate_finger_angles(self, hand_lms):
        if np.sum(hand_lms) == 0:
            return [0.0] * 15
        angles = []
        indices = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
        for p1_idx, p2_idx, p3_idx in indices:
            angles.append(calculate_angle_3d(hand_lms[p1_idx], hand_lms[p2_idx], hand_lms[p3_idx]))
        return angles

    def _calculate_palm_normal(self, hand_lms):
        if np.sum(hand_lms) == 0:
            return np.zeros(3)
        p0, p5, p17 = hand_lms[0], hand_lms[5], hand_lms[17]
        v1, v2 = p5 - p0, p17 - p0
        palm_normal = np.cross(v1, v2)
        norm = np.linalg.norm(palm_normal)
        return palm_normal / norm if norm > 1e-6 else np.zeros(3)

    def _calculate_relational_features(self, pose_lms, left_hand_lms, right_hand_lms):
        features = []
        nose = pose_lms[0] if pose_lms.shape[0] > 0 else np.zeros(3)
        left_shoulder = pose_lms[1] if pose_lms.shape[0] > 1 else np.zeros(3)
        right_shoulder = pose_lms[2] if pose_lms.shape[0] > 2 else np.zeros(3)
        left_wrist = left_hand_lms[0] if np.sum(left_hand_lms) != 0 else np.zeros(3)
        right_wrist = right_hand_lms[0] if np.sum(right_hand_lms) != 0 else np.zeros(3)

        features.append(np.linalg.norm(left_wrist - right_wrist))
        features.append(np.linalg.norm(right_wrist - nose))
        features.append(np.linalg.norm(left_wrist - nose))
        features.append(np.linalg.norm(right_wrist - left_shoulder))
        features.append(np.linalg.norm(left_wrist - right_shoulder))
        return np.array(features)

    def _calculate_all_features(self, pose_seq, left_hand_seq, right_hand_seq, prev_data):
        features_seq = []
        prev_velocity = prev_data.get('velocity') if prev_data else None
        num_coords = (pose_seq.shape[1] + left_hand_seq.shape[1] + right_hand_seq.shape[1]) * 3

        for i in range(len(pose_seq)):
            frame_features = []
            if i > 0:
                pose_disp = pose_seq[i] - pose_seq[i-1]
                left_hand_disp = left_hand_seq[i] - left_hand_seq[i-1]
                right_hand_disp = right_hand_seq[i] - right_hand_seq[i-1]
                velocity = np.concatenate((pose_disp.flatten(), left_hand_disp.flatten(), right_hand_disp.flatten()))
                acceleration = velocity - prev_velocity if prev_velocity is not None else np.zeros_like(velocity)
                prev_velocity = velocity
            else:
                velocity = np.zeros(num_coords)
                acceleration = np.zeros_like(velocity)
                prev_velocity = velocity

            frame_features.extend(velocity)
            frame_features.extend(acceleration)

            p_frame, l_frame, r_frame = pose_seq[i], left_hand_seq[i], right_hand_seq[i]
            frame_features.extend(self._calculate_finger_angles(l_frame))
            frame_features.extend(self._calculate_finger_angles(r_frame))
            frame_features.extend(self._calculate_palm_normal(l_frame))
            frame_features.extend(self._calculate_palm_normal(r_frame))
            frame_features.extend(self._calculate_relational_features(p_frame, l_frame, r_frame))

            frame_features = np.array(frame_features, dtype=np.float32)
            if self.scaler is not None:
                non_zero_mask = np.abs(frame_features).sum() > 0
                if non_zero_mask:
                    frame_features = self.scaler.transform(frame_features.reshape(1, -1)).flatten()
            features_seq.append(frame_features)

        current_data = {
            'pose': pose_seq[-1] if len(pose_seq) > 0 else np.zeros((9, 3)),
            'left': left_hand_seq[-1] if len(left_hand_seq) > 0 else np.zeros((21, 3)),
            'right': right_hand_seq[-1] if len(right_hand_seq) > 0 else np.zeros((21, 3)),
            'velocity': prev_velocity
        }
        return np.array(features_seq), current_data

    def get_combined_features(self, image, prev_data):
        pose_lms, left_lms, right_lms, hands_results = self._extract_landmarks(image)
        norm_pose, norm_left, norm_right = self._normalize_landmarks(pose_lms, left_lms, right_lms)
        features_seq, current_data = self._calculate_all_features(
            np.array([norm_pose]), np.array([norm_left]), np.array([norm_right]), prev_data
        )
        return features_seq[0], current_data, hands_results

    def close(self):
        self.pose.close()
        self.hands.close()