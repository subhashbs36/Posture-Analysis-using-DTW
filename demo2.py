import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer 
import cv2
import time
import pose_module as pm
from scipy.spatial.distance import cosine
from PyQt5.QtGui import QImage, QPixmap
from fastdtw import fastdtw
import numpy as np

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, benchmark_video, user_video, fps_limit=60):
        super().__init__()
        self.benchmark_video = benchmark_video
        self.user_video = user_video
        self.fps_limit = fps_limit
        self.stopped = False
        self.pausing_video = False
        self.analyzing = True  # Set analyzing to True by default
        
    def run(self):
        benchmark_cam = cv2.VideoCapture(self.benchmark_video)
        user_cam = cv2.VideoCapture(self.user_video)

        if not benchmark_cam.isOpened():
            print(f"Error: Could not open benchmark video at {self.benchmark_video}")
            return

        if not user_cam.isOpened():
            print(f"Error: Could not open user video at {self.user_video}")
            return

        fps_time = 0  # Initializing fps to 0

        detector_1 = pm.poseDetector()
        detector_2 = pm.poseDetector()
        frame_counter = 0
        correct_frames = 0

        while (benchmark_cam.isOpened() or user_cam.isOpened()) and not self.stopped:
            try:
                ret_val, image_1 = user_cam.read()
                if frame_counter == user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    correct_frames = 0
                    user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

                image_1 = cv2.resize(image_1, (720, 640))
                image_1 = detector_1.findPose(image_1)
                lmList_user = detector_1.findPosition(image_1, draw=False)

                ret_val_1, image_2 = benchmark_cam.read()
                if frame_counter == benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    correct_frames = 0
                    benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

                image_2 = cv2.resize(image_2, (720, 640))
                image_2 = detector_2.findPose(image_2)
                lmList_benchmark = detector_2.findPosition(image_2, draw=False)

                frame_counter += 1

                if ret_val_1 or ret_val:
                    error = 0
                    if self.analyzing:
                        error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)

                    cv2.putText(image_1, 'Error: {}%'.format(str(round(100 * (float(error)), 2))), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if error < 0.3:
                        cv2.putText(image_1, "CORRECT STEPS", (40, 600),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        correct_frames += 1
                    else:
                        cv2.putText(image_1, "INCORRECT STEPS", (40, 600),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if frame_counter == 0:
                        frame_counter = user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
                    cv2.putText(image_1, "Steps Accurately Done: {}%".format(
                        str(round(100 * correct_frames / frame_counter, 2))), (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    if self.analyzing:
                        concatenated_image = np.concatenate((image_2, image_1), axis=1)
                        self.change_pixmap_signal.emit(concatenated_image)

                    fps_time = time.time()
                    time.sleep(1 / self.fps_limit)
                else:
                    break
            except:
                pass

        benchmark_cam.release()
        user_cam.release()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.video_thread = None

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.analyze_button = QPushButton('Analyze/Stop', self)
        self.analyze_button.clicked.connect(self.toggle_analysis)
        self.layout.addWidget(self.analyze_button)

        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(60)
        self.fps_slider.valueChanged.connect(self.update_fps_limit)
        self.layout.addWidget(self.fps_slider)

        self.setLayout(self.layout)
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Dance Comparison')

        # Add a delay before starting the video thread
        self.delay_start_video('Fast Bowling Trimed.mp4', 'Fast Bowling Trimed.mp4')

        self.show()

    def toggle_analysis(self):
        if self.video_thread:
            self.video_thread.analyzing = not self.video_thread.analyzing
            if self.video_thread.analyzing:
                self.analyze_button.setText('Stop Analysis')
            else:
                self.analyze_button.setText('Analyze')

    def update_fps_limit(self, value):
        if self.video_thread:
            self.video_thread.fps_limit = value

    def delay_start_video(self, benchmark_video, user_video):
        # Add a delay before starting the video thread
        QTimer.singleShot(1000, lambda: self.start_video(benchmark_video, user_video))

    def start_video(self, benchmark_video, user_video):
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(benchmark_video, user_video, self.fps_slider.value())
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stopped = True
            self.video_thread.wait()

    def update_image(self, image):
        q_image = self.convert_np_to_qimage(image)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def convert_np_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


