from re import I
import time
start = time.time()
import cv2
import json, os, csv, math, logging
import logging
import math
import numpy as np
from squash.Player import Player
from ultralytics import YOLO
from matplotlib import pyplot as plt
from squash.Ball import Ball
import sys
import csv
print(f"time to import everything: {time.time()-start}")
alldata = organizeddata = []
def get_reference_points(path, frame_width, frame_height):
    # Mouse callback function to capture click events
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append((x, y))
            print(f"Point captured: ({x}, {y})")
            cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Court", frame1)

    # Function to save reference points to a file
    def save_reference_points(file_path):
        with open(file_path, "w") as f:
            json.dump(reference_points, f)
        print(f"reference points saved to {file_path}")

    # Function to load reference points from a file
    def load_reference_points(file_path):
        global reference_points
        with open(file_path, "r") as f:
            reference_points = json.load(f)
        print(f"reference points loaded from {file_path}")

    # Load the frame (replace 'path_to_frame' with the actual path)
    if os.path.isfile("reference_points.json"):
        load_reference_points("reference_points.json")
        print(f"Loaded reference points: {reference_points}")
        return reference_points
    else:
        print(
            "No reference points file found. Please click on the court to set reference points."
        )
        frame_count = 1
        cap2 = cv2.VideoCapture(path)
        frame1 = None
        while cap2.isOpened():
            success, frame = cap2.read()
            if not success:
                break
            frame_count += 1
            if frame_count == 100:
                frame1 = frame
                break
        frame1 = cv2.resize(frame1, (frame_width, frame_height))
        cv2.imshow("Court", frame1)
        cv2.setMouseCallback("Court", click_event)

        print(
            "Click on the key points of the court. Press 's' to save and 'q' to quit.\nMake sure to click in the following order shown by the example"
        )
        example_image = cv2.imread("output/annotated-squash-court.png")
        example_image_resized = cv2.resize(example_image, (frame_width, frame_height))
        cv2.imshow("Court Example", example_image_resized)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_reference_points("reference_points.json")
                cv2.destroyAllWindows()
                return reference_points
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return reference_points


def find_match_2d_array(array, x):
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False
def drawmap(lx, ly, rx, ry, map):
    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), map.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), map.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), map.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), map.shape[0] - 1)
    map[ly, lx] += 1
    map[ry, rx] += 1

def cleanwrite(home_path):
    #if not os.path.exists(home_path), then create the file
    if not os.path.exists(home_path):
        os.makedirs(home_path, exist_ok=True)
    
    with open(f"{home_path}/ball.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/player1.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/player2.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/ball-xyn.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/read_ball.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/read_player1.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/read_player2.txt", "w") as f:
        f.write("")
    with open(f"{home_path}/final.json", "w") as f:
        f.write("[")
    with open(f"{home_path}/final.csv", "w") as f:
        f.write(
            "Frame count,Player 1 Keypoints,Player 2 Keypoints,Ball Position,Shot Type,Player 1 RL World Position,Player 2 RL World Position,Ball RL World Position,Who Hit the Ball\n"
        )


def find_last(i, otherTrackIds):
    possibleits = []
    for it in range(len(otherTrackIds)):
        if otherTrackIds[it][1] == i:
            possibleits.append(it)
    return possibleits[-1]
def create_court(court_width=400, court_height=610):
    # Create a blank image
    court = np.zeros((court_height, court_width, 3), np.uint8)
    t_point=(int(court_width/2), int(0.5579487*court_height))
    left_service_box_end=(0, int(0.5579487*court_height))
    right_service_box_end=(int(court_width), int(0.5579487*court_height))
    left_service_box_start=(int(0.25*court_width), int(0.5579487*court_height))
    right_service_box_start=(int(0.75*court_width), int(0.5579487*court_height))
    left_service_box_low=(int(0.25*court_width), int(0.7220512*court_height))
    right_service_box_low=(int(0.75*court_width), int(0.7220512*court_height))
    left_service_box_low_end=(0, int(0.7220512*court_height))
    right_service_box_low_end=(int(court_width), int(0.7220512*court_height))
    points=[t_point, left_service_box_end, right_service_box_end, left_service_box_start, right_service_box_start, left_service_box_low, right_service_box_low, left_service_box_low_end, right_service_box_low_end]
    #plot all points
    for point in points:
        cv2.circle(court, point, 5, (255, 255, 255), -1)
    #between service boxes
    cv2.line(court, left_service_box_end, right_service_box_end, (255, 255, 255), 2)
    #between t point and middle low
    cv2.line(court, t_point, (int(court_width/2), court_height), (255, 255, 255), 2)
    #between low service boxes
    #cv2.line(court, left_service_box_low_end, right_service_box_low_end, (255, 255, 255), 2)
    #betwwen service boxes and low service boxes
    cv2.line(court, left_service_box_end, left_service_box_low_end, (255, 255, 255), 2)
    cv2.line(court, right_service_box_end, right_service_box_low_end, (255, 255, 255), 2)
    cv2.line(court, left_service_box_start, left_service_box_low, (255, 255, 255), 2)
    cv2.line(court, right_service_box_start, right_service_box_low, (255, 255, 255), 2)
    cv2.line(court, left_service_box_low_end, left_service_box_low, (255, 255, 255), 2)
    cv2.line(court, right_service_box_low_end, right_service_box_low, (255, 255, 255), 2)
    return court
def visualize_court():
    court = create_court()
    cv2.imshow('Squash Court', court)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def plot_on_court(court, p1, p2):
    #draw p1 in red and p2 in blue
    cv2.circle(court, p1, 5, (0, 0, 255), -1)
    cv2.circle(court, p2, 5, (255, 0, 0), -1)
    return court
def generate_2d_homography(reference_points_px, reference_points_3d):
    """
    generate a homography matrix for 2d to 2d mapping(for the top down court view)
    """
    try:
        # Convert to numpy arrays
        reference_points_px = np.array(reference_points_px, dtype=np.float32)
        reference_points_3d = np.array(reference_points_3d, dtype=np.float32)

        # Calculate homography matrix
        H, _ = cv2.findHomography(reference_points_px, reference_points_3d)

        return H

    except Exception as e:
        print(f"Error generating 2D homography: {str(e)}")
        return None
def to_court_px(player1pos, player2pos, homography):
    """
    given player positions in the frame, convert to top down court positions
    """
    try:
        # Convert to numpy arrays
        player1pos = np.array(player1pos, dtype=np.float32)
        player2pos = np.array(player2pos, dtype=np.float32)

        # Apply homography transformation
        player1_court = cv2.perspectiveTransform(player1pos.reshape(-1, 1, 2), homography)
        player2_court = cv2.perspectiveTransform(player2pos.reshape(-1, 1, 2), homography)

        return player1_court, player2_court

    except Exception as e:
        print(f"Error converting to court positions: {str(e)}")
        return None, None

def load_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()

    # Convert the data to a list of floats
    data = [float(line.strip()) for line in data]

    # Group the data into pairs of coordinates (x, y)
    positions = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

    return positions

def main(path="main_laptop.mp4", frame_width=640, frame_height=360, output_path="main_research"):
    try:
        print("imported all")
        csvstart = 0
        end = csvstart + 100
        cleanwrite(home_path=output_path)
        pose_model = YOLO("models/yolo11n-pose.pt")
        ballmodel = YOLO("trained-models\\g-ball2(white_latest).pt")
        print("loaded models")
        cap = cv2.VideoCapture(path)
        with open(f"{output_path}/final.txt", "w") as f:
            f.write(
                f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n"
            )
        players = {}
        courtref = 0
        occlusion_times = {}
        for i in range(1, 3):
            occlusion_times[i] = 0
        future_predict = None
        player_last_positions = {}
        frame_count = 0
        ball_false_pos = []
        past_ball_pos = []
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        video_out_path = output_path+"/annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        importantoutputpath = output_path+"/important.mp4"
        cv2.VideoWriter(importantoutputpath, fourcc, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))
        detections = []
        plast=[[],[]]
        mainball = Ball(0, 0, 0, 0)
        ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        otherTrackIds = [[0, 0], [1, 1], [2, 2]]
        updated = [[False, 0], [False, 0]]
        reference_points = []
        reference_points = get_reference_points(
            path=path, frame_width=frame_width, frame_height=frame_height
        )
        is_rally=False
        references1 = []
        references2 = []

        pixdiffs = []

        p1distancesfromT = []
        p2distancesfromT = []

        courtref = np.int64(courtref)
        referenceimage = None

        # function to see what kind of shot has been hit

        reference_points_3d = [
            [0, 9.75, 0],  # Top-left corner, 1
            [6.4, 9.75, 0],  # Top-right corner, 2
            [6.4, 0, 0],  # Bottom-right corner, 3
            [0, 0, 0],  # Bottom-left corner, 4
            [3.2, 4.57, 0],  # "T" point, 5
            [0, 2.71, 0],  # Left bottom of the service box, 6
            [6.4, 2.71, 0],  # Right bottom of the service box, 7
            [0, 9.75, 0.48],  # left of tin, 8
            [6.4, 9.75, 0.48],  # right of tin, 9
            [0, 9.75, 1.78],  # Left of the service line, 10
            [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
            [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
        ]
        
        court_view=create_court()
        #visualize_court()
        np.zeros((frame_height, frame_width), dtype=np.float32)
        np.zeros((frame_height, frame_width), dtype=np.float32)
        #create a white image that is frame_height x frame_width
        npheatmap_image=np.zeros((frame_height, frame_width, 3), dtype=np.float32)
        heatmap_image=create_court(court_width=400, court_height=610)    
        if heatmap_image is None:
            raise FileNotFoundError(
                f"Could not generate heatmap overlay "
            )
        np.zeros_like(heatmap_image, dtype=np.float32)

        ballxy = []
        with open(f'{output_path}/reference_points.csv', 'w', newline='') as file:
            writer=csv.writer(file)
            writer.writerow(reference_points)
            writer.writerow(reference_points_3d)
        print("saved reference points")
        
        running_frame = 0
        print("started video input")
        print(f"loaded everything in {time.time()-start} seconds")
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            # force it to go to lowestx-->highestx and then lowesty-->highesty
            frame_count += 1

            if len(references1) != 0 and len(references2) != 0:
                sum(references1) / len(references1)
                sum(references2) / len(references2)

            running_frame += 1
            
            ball = ballmodel(frame)

            annotated_frame = frame.copy() 

            for reference in reference_points:
                cv2.circle(
                    annotated_frame,
                    (int(reference[0]), int(reference[1])),
                    5,
                    (0, 255, 0),
                    2,
                )

            try:
                highestconf = 0
                x1 = x2 = y1 = y2 = 0
                # Ball detection
                ball = ballmodel(frame)
                label = ""
                try:
                    # Get the last 10 positions
                    start_idx = max(0, len(past_ball_pos) - 10)
                    positions = past_ball_pos[start_idx:]
                    # Loop through positions
                    for i in range(len(positions)):
                        pos = positions[i]
                        # Draw circle with constant radius
                        radius = 5  # Adjust as needed
                        cv2.circle(
                            annotated_frame, (pos[0], pos[1]), radius, (255, 255, 255), 2
                        )
                        # Draw line to next position
                        if i < len(positions) - 1:
                            next_pos = positions[i + 1]
                            cv2.line(
                                annotated_frame,
                                (pos[0], pos[1]),
                                (next_pos[0], next_pos[1]),
                                (255, 255, 255),
                                2,
                            )
                except Exception:
                    pass
                for box in ball[0].boxes:
                    coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
                    x1temp, y1temp, x2temp, y2temp = coords
                    label = ballmodel.names[int(box.cls)]
                    confidence = float(box.conf)  # Convert tensor to float
                    int((x1temp + x2temp) / 2)
                    int((y1temp + y2temp) / 2)

                    if confidence > highestconf:
                        highestconf = confidence
                        x1 = x1temp
                        y1 = y1temp
                        x2 = x2temp
                        y2 = y2temp

                cv2.rectangle(
                    annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )

                cv2.putText(
                    annotated_frame,
                    f"{label} {highestconf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # print(label)
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                avg_x = int((x1 + x2) / 2)
                avg_y = int((y1 + y2) / 2)
                size = avg_x * avg_y
                if avg_x > 0 or avg_y > 0:
                    if mainball.getlastpos()[0] != avg_x or mainball.getlastpos()[1] != avg_y:
                        mainball.update(avg_x, avg_y, size)
                        past_ball_pos.append([avg_x, avg_y, running_frame])
                        math.hypot(
                            avg_x - mainball.getlastpos()[0], avg_y - mainball.getlastpos()[1]
                        )
                        drawmap(
                            mainball.getloc()[0],
                            mainball.getloc()[1],
                            mainball.getlastpos()[0],
                            mainball.getlastpos()[1],
                            ballmap,
                        )
            
                """
                FRAMEPOSE
                """
                try:
                    track_results = pose_model.track(frame, persist=True, show=False)
                    if (
                        track_results
                        and hasattr(track_results[0], "keypoints")
                        and track_results[0].keypoints is not None
                    ):
                        # Extract boxes, track IDs, and keypoints from pose results
                        boxes = track_results[0].boxes.xywh.cpu()
                        track_ids = track_results[0].boxes.id.int().cpu().tolist()
                        keypoints = track_results[0].keypoints.cpu().numpy()
                        set(track_ids)
                        # Update or add players for currently visible track IDs
                        # note that this only works with occluded players < 2, still working on it :(
                        #print(f"number of players found: {len(track_ids)}")
                        # occluded as [[players_found, last_pos_p1, last_pos_p2, frame_number]...]
                        if len(track_ids) < 2:
                            print(player_last_positions)
                            last_pos_p1 = player_last_positions.get(1, (None, None))
                            last_pos_p2 = player_last_positions.get(2, (None, None))
                            # print(f"last pos p1: {last_pos_p1}")
                            occluded = []
                            try: 
                                occluded.append( 
                                    [
                                        len(track_ids),
                                        last_pos_p1,
                                        last_pos_p2,
                                        frame_count,
                                    ]
                                )
                            except Exception:
                                pass
                        if len(track_ids) > 2:
                            print(f"track ids were greater than 2: {track_ids}")
                            continue

                        for box, track_id, kp in zip(boxes, track_ids, keypoints):
                            x, y, w, h = box
                            player_crop = frame[int(y) : int(y + h), int(x) : int(x + w)]

                            if player_crop.size == 0:
                                continue
                            if not find_match_2d_array(otherTrackIds, track_id):
                                # player 1 has been updated last
                                if updated[0][1] > updated[1][1]:
                                    if len(references2) > 1:
                                        otherTrackIds.append([track_id, 2])
                                        print(f"added track id {track_id} to player 2")
                                else:
                                    otherTrackIds.append([track_id, 1])
                                    print(f"added track id {track_id} to player 1")
                            # if updated[0], then that means that player 1 was updated last
                            # bc of this, we can assume that the next player is player 2
                            
                            if track_id == 1:
                                playerid = 1
                            elif track_id == 2:
                                playerid = 2
                            # updated [0] is player 1, updated [1] is player 2
                            # if player1 was updated last, then player 2 is next
                            # if player 2 was updated last, then player 1 is next
                            # if both were updated at the same time, then player 1 is next as track ids go from 1 --> 2 im really hoping
                            elif updated[0][1] > updated[1][1]:
                                playerid = 2
                                # player 1 was updated last
                            elif updated[0][1] < updated[1][1]:
                                playerid = 1
                                # player 2 was updated last
                            elif updated[0][1] == updated[1][1]:
                                playerid = 1
                                # both players were updated at the same time, so we are assuming that player 1 is the next player
                            else:
                                print(f"could not find player id for track id {track_id}")
                                continue
                            # If player is already tracked, update their info
                            if playerid in players:
                                players[playerid].add_pose(kp)
                                player_last_positions[playerid] = (x, y)  # Update position
                                players[playerid].add_pose(kp)
                                if playerid == 1:
                                    updated[0][0] = True
                                    updated[0][1] = frame_count
                                if playerid == 2:
                                    updated[1][0] = True
                                    updated[1][1] = frame_count
                            if len(players) < 2:
                                players[otherTrackIds[track_id][0]] = Player(
                                    player_id=otherTrackIds[track_id][1]
                                )
                                player_last_positions[playerid] = (x, y)
                                if playerid == 1:
                                    updated[0][0] = True
                                    updated[0][1] = frame_count
                                else:
                                    updated[1][0] = True
                                    updated[1][1] = frame_count
                                print(f"Player {playerid} added.")
                            # putting player keypoints on the frame
                            pos=[]
                            for keypoint in kp:
                                # print(keypoint.xyn[0])
                                i = 0
                                for k in keypoint.xyn[0]:
                                    x, y = k
                                    x = int(x * frame_width)
                                    y = int(y * frame_height)
                                    if playerid == 1:
                                        cv2.circle(
                                            annotated_frame, (int(x), int(y)), 3, (0, 0, 255), 5
                                        )
                                    else:
                                        cv2.circle(
                                            annotated_frame, (int(x), int(y)), 3, (255, 0, 0), 5
                                        )
                                    if i == 16:
                                        cv2.putText(
                                            annotated_frame,
                                            f"{playerid}",
                                            (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            2.5,
                                            (255, 255, 255),
                                            7,
                                        )
                                        pos.append([x,y])
                                    if i == 15:
                                        pos.append([x,y])
                                    if i==10:
                                        cv2.putText(
                                            annotated_frame,
                                            f"{track_id}",
                                            (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (255, 255, 255),
                                            2,
                                        )
                                    i += 1
                            #print(f'playerid was :{playerid}')
                            if len(pos)==2:
                                avgpos=[(pos[0][0]+pos[1][0])/2,(pos[0][1]+pos[1][1])/2, frame_count]
                                if playerid==1:
                                    plast[0].append(avgpos)
                                else:
                                    plast[1].append(avgpos)
                            elif len(pos)==1:
                                pos.append(frame_count)
                                if playerid==1:
                                    plast[0].append(pos)
                                else:
                                    plast[1].append(pos)
                            else:
                                #print(f'kp: {kp}')
                                print(f"could not find the player position for player {playerid}")
                        
                        # in the form of [ball shot type, player1 proximity to the ball, player2 proximity to the ball, ]
                        importantdata = []
                except Exception as e:
                    print(f"got error in framepose: {e}")
                    print(f"line number was {sys.exc_info()[-1].tb_lineno}")
                    continue
            except Exception as e:
                print(f'got error in ballplayer detections: {e}')
                print(f'line number was {sys.exc_info()[-1].tb_lineno}')
                continue
            if players.get(1) and players.get(2) is not None:
                if (
                    players.get(1).get_latest_pose()
                    or players.get(2).get_latest_pose() is not None
                ):
                    # print('line 265')
                    try:
                        p1_left_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p1_left_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p1_right_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p1_right_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][15][1]
                            * frame_height
                        )
                    except Exception:
                        p1_left_ankle_x = p1_left_ankle_y = p1_right_ankle_x = (
                            p1_right_ankle_y
                        ) = 0
                    try:
                        p2_left_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p2_left_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p2_right_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p2_right_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][15][1]
                            * frame_height
                        )
                    except Exception:
                        p2_left_ankle_x = p2_left_ankle_y = p2_right_ankle_x = (
                            p2_right_ankle_y
                        ) = 0
                    # Display the ankle positions on the bottom left of the frame
                    avgxank1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgyank1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgxank2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgyank2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    text_p1 = f"P1 position(ankle): {avgxank1},{avgyank1}"
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[find_last(1, otherTrackIds)][1]}",
                        (p1_left_ankle_x, p1_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[find_last(2, otherTrackIds)][1]}",
                        (p2_left_ankle_x, p2_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    text_p2 = f"P2 position(ankle): {avgxank2},{avgyank2}"
                    cv2.putText(
                        annotated_frame,
                        text_p1,
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2,
                        (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    avgpx1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgpy1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgpx2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgpy2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    # print(reference_points)
                    p1distancefromT = math.hypot(
                        reference_points[4][0] - avgpx1, reference_points[4][1] - avgpy1
                    )
                    p2distancefromT = math.hypot(
                        reference_points[4][0] - avgpx2, reference_points[4][1] - avgpy2
                    )
                    p1distancesfromT.append(p1distancefromT)
                    p2distancesfromT.append(p2distancefromT)
                    text_p1t = f"P1 distance from T: {p1distancesfromT[-1]}"
                    text_p2t = f"P2 distance from T: {p2distancesfromT[-1]}"
                    cv2.putText(
                        annotated_frame,
                        text_p1t,
                        (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2t,
                        (10, frame_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
            try:
                # Generate player ankle heatmap
                if (
                    players.get(1).get_latest_pose() is not None
                    and players.get(2).get_latest_pose() is not None
                ):
                    player_ankles = [
                        (
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][0]
                                * frame_width
                            ),
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][1]
                                * frame_height
                            ),
                        ),
                        (
                            int(
                                players.get(2).get_latest_pose().xyn[0][16][0]
                                * frame_width
                            ),
                            int(
                                players.get(2).get_latest_pose().xyn[0][16][1]
                                * frame_height
                            ),
                        ),
                    ]

                    # Draw points on the heatmap
                    for ankle in player_ankles:
                        cv2.circle(
                            heatmap_image, ankle, 5, (255, 0, 0), -1
                        )  # Blue points for Player 1
                        cv2.circle(
                            heatmap_image, ankle, 5, (0, 0, 255), -1
                        )  # Red points for Player 2

                blurred_heatmap_ankle = cv2.GaussianBlur(heatmap_image, (51, 51), 0)

                # Normalize heatmap and apply color map in one step
                normalized_heatmap = cv2.normalize(
                    blurred_heatmap_ankle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )
                heatmap_overlay = cv2.applyColorMap(
                    normalized_heatmap, cv2.COLORMAP_JET
                )

                # Combine with white image
                cv2.addWeighted(
                    np.ones_like(heatmap_overlay) * 255, 0.5, heatmap_overlay, 0.5, 0
                )
            except Exception:
                # print(f"line618: {e}")
                pass
            # Save the combined image
            # cv2.imwrite("output/heatmap_ankle.png", combined_image)
            ballx = bally = 0
            # ball stuff
            if (
                mainball is not None
                and mainball.getlastpos() is not None
                and mainball.getlastpos() != (0, 0)
            ):
                ballx = mainball.getlastpos()[0]
                bally = mainball.getlastpos()[1]
                if ballx != 0 and bally != 0:
                    if [ballx, bally] not in ballxy:
                        ballxy.append([ballx, bally, frame_count])
                        # print(
                        #     f"ballx: {ballx}, bally: {bally}, appended to ballxy with length {len(ballxy)} and frame count as : {frame_count}"
                        # )

            # Draw the ball trajectory
            if len(ballxy) > 2:
                for i in range(1, len(ballxy)):
                    if ballxy[i - 1] is None or ballxy[i] is None:
                        continue
                    if ballxy[i][2] - ballxy[i - 1][2] < 7:
                        if frame_count - ballxy[i][2] < 7:
                            cv2.line(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                (ballxy[i][0], ballxy[i][1]),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i][0], ballxy[i][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )


            for ball_pos in ballxy:
                if frame_count - ball_pos[2] < 7:
                    # print(f'wrote to frame on line 1028 with coords: {ball_pos}')
                    cv2.circle(
                        annotated_frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 0), -1
                    )

            #dump all data
            def dump(path, player1kp, player2kp, ballpos):
                data = [
                    frame_count,
                    player1kp,
                    player2kp,
                    ballpos,
                ]
                #write to a csv file
                with open(path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
            try:
                dump(f'{output_path}/final.csv', players.get(1).get_latest_pose().xyn, players.get(2).get_latest_pose().xyn, [ballx, bally])
                print(f'ballxy was {plast[0]}')
            except Exception as e:
                print(f"error in dumping data: {e}")
                pass
            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)

            #print(f"finished frame {frame_count}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"error2: {e}")
        print(f"line was {e.__traceback__.tb_lineno}")
        print(f"other into about e: {e.__traceback__}")
        print(f"other info about e: {e.__traceback__.tb_frame}")
        print(f"other info about e: {e.__traceback__.tb_next}")
        print(f"other info about e: {e.__traceback__.tb_lasti}")


if __name__ == "__main__":
    try:
        main(path="C:/Users/default.DESKTOP-7FKFEEG/Downloads/Farag v Coll 2019 (from squashtv).mp4", output_path='farag v coll 2019 squashtv out')
    # get keyboarinterrupt error
    except KeyboardInterrupt:
        print("keyboard interrupt")

        exit()
    except Exception:
        # print(f"error: {e}")
        pass



