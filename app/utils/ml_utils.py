import cv2
import mediapipe as mp
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return angle

def calculate_kinematic_features(pose_seq, left_hand_seq, right_hand_seq, prev_pose, prev_left, prev_right):
    frame_features = []
    
    pose_disp = pose_seq - prev_pose if prev_pose is not None else np.zeros_like(pose_seq)
    left_hand_disp = left_hand_seq - prev_left if prev_left is not None else np.zeros_like(left_hand_seq)
    right_hand_disp = right_hand_seq - prev_right if prev_right is not None else np.zeros_like(right_hand_seq)
    frame_features.extend(pose_disp.flatten())
    frame_features.extend(left_hand_disp.flatten())
    frame_features.extend(right_hand_disp.flatten())
    
    if np.sum(pose_seq[2]) != 0 and np.sum(pose_seq[4]) != 0 and np.sum(pose_seq[6]) != 0:
        angle_right_elbow = calculate_angle(pose_seq[2], pose_seq[4], pose_seq[6])
        frame_features.append(angle_right_elbow)
    else:
        frame_features.append(0.0)

    if np.sum(pose_seq[1]) != 0 and np.sum(pose_seq[3]) != 0 and np.sum(pose_seq[5]) != 0:
        angle_left_elbow = calculate_angle(pose_seq[1], pose_seq[3], pose_seq[5])
        frame_features.append(angle_left_elbow)
    else:
        frame_features.append(0.0)

    def _calculate_finger_angles(hand_lms):
        if np.sum(hand_lms) == 0: return [0.0] * 15
        angles = []
        finger_lms_indices = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            (0, 9, 10), (9, 10, 11), (10, 11, 12),
            (0, 13, 14), (13, 14, 15), (14, 15, 16),
            (0, 17, 18), (17, 18, 19), (18, 19, 20)
        ]
        for p1_idx, p2_idx, p3_idx in finger_lms_indices:
            angles.append(calculate_angle(hand_lms[p1_idx], hand_lms[p2_idx], hand_lms[p3_idx]))
        return angles
    
    frame_features.extend(_calculate_finger_angles(left_hand_seq))
    frame_features.extend(_calculate_finger_angles(right_hand_seq))

    def _calculate_palm_vector(hand_lms):
        if np.sum(hand_lms) == 0: return [0.0, 0.0, 0.0]
        wrist = hand_lms[0]
        palm_middle = (hand_lms[9] + hand_lms[13]) / 2
        vector = palm_middle - wrist
        norm = np.linalg.norm(vector)
        return list(vector / norm) if norm != 0 else [0.0, 0.0, 0.0]

    frame_features.extend(_calculate_palm_vector(left_hand_seq))
    frame_features.extend(_calculate_palm_vector(right_hand_seq))

    if np.sum(pose_seq[0]) != 0 and np.sum(left_hand_seq[0]) != 0:
        dist_left_hand_to_nose = np.linalg.norm(left_hand_seq[0] - pose_seq[0])
        frame_features.append(dist_left_hand_to_nose)
    else:
        frame_features.append(0.0)
    
    if np.sum(pose_seq[0]) != 0 and np.sum(right_hand_seq[0]) != 0:
        dist_right_hand_to_nose = np.linalg.norm(right_hand_seq[0] - pose_seq[0])
        frame_features.append(dist_right_hand_to_nose)
    else:
        frame_features.append(0.0)
    
    if np.sum(pose_seq[5]) != 0 and np.sum(pose_seq[6]) != 0:
        dist_wrists = np.linalg.norm(pose_seq[5] - pose_seq[6])
        frame_features.append(dist_wrists)
    else:
        frame_features.append(0.0)
    
    return np.array(frame_features, dtype=np.float32)

def extract_all_features(image, prev_pose, prev_left, prev_right):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    pose_landmarks = np.zeros((7, 3))
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        selected_indices = [0, 11, 12, 13, 14, 15, 16]
        for i, idx in enumerate(selected_indices):
            pose_landmarks[i] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    left_hand_lms = np.zeros((21, 3))
    right_hand_lms = np.zeros((21, 3))
    if hands_results.multi_hand_landmarks:
        for hand_lm_obj in hands_results.multi_hand_landmarks:
            hand_lms_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm_obj.landmark])
            hand_label = hands_results.multi_handedness[hands_results.multi_hand_landmarks.index(hand_lm_obj)].classification[0].label
            
            if hand_label == "Right":
                right_hand_lms = hand_lms_array
            else:
                left_hand_lms = hand_lms_array
                
    left_shoulder = pose_landmarks[1]
    right_shoulder = pose_landmarks[2]
    if np.sum(left_shoulder) == 0:
        center_shoulder = right_shoulder
    elif np.sum(right_shoulder) == 0:
        center_shoulder = left_shoulder
    else:
        center_shoulder = (left_shoulder + right_shoulder) / 2.0
    
    if np.sum(center_shoulder) != 0:
        norm_pose_lms = pose_landmarks - center_shoulder
        norm_left_hand_lms = left_hand_lms - center_shoulder
        norm_right_hand_lms = right_hand_lms - center_shoulder
    else:
        norm_pose_lms = pose_landmarks
        norm_left_hand_lms = left_hand_lms
        norm_right_hand_lms = right_hand_lms

    raw_landmarks = np.concatenate([
        norm_pose_lms.flatten(),
        norm_left_hand_lms.flatten(),
        norm_right_hand_lms.flatten()
    ])

    kinematic_features = calculate_kinematic_features(
        norm_pose_lms, norm_left_hand_lms, norm_right_hand_lms,
        prev_pose, prev_left, prev_right
    )

    all_features = np.concatenate([raw_landmarks, kinematic_features])

    return all_features, norm_pose_lms, norm_left_hand_lms, norm_right_hand_lms, pose_results, hands_results