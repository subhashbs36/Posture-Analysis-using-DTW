import cv2
import time
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from pose_module import poseDetector
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PoseComparisonApp(QMainWindow):
    def __init__(self, benchmark_video, user_video):
        super(PoseComparisonApp, self).__init__()

        self.benchmark_cam = cv2.VideoCapture(benchmark_video)
        self.user_cam = cv2.VideoCapture(user_video)
        self.detector_1 = poseDetector()
        self.detector_2 = poseDetector()
        self.frame_counter = 0
        self.correct_frames = 0
        self.fps_time = 0
        self.paused = False
        self.errors_user = []  # List to store errors for the user video
        self.errors_benchmark = []  # List to store errors for the benchmark video
        self.accuracy = []  

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        title_label = QLabel("<b>Bowling Comparison</b>", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px")  # You can adjust the font size

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(60)
        self.slider.setValue(60)
        self.slider.valueChanged.connect(self.update_fps)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(title_label)  # Add the title label to the layout
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.pause_button)

        # Matplotlib integration
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(121)  # 1 row, 2 columns, first subplot
        self.ax.set_title("User and Benchmark Error")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Error")

        self.ax_difference = self.figure.add_subplot(122)  # 1 row, 2 columns, second subplot
        self.ax_difference.set_title("Error Difference")
        self.ax_difference.set_xlabel("Frame")
        self.ax_difference.set_ylabel("Error Difference")

        # Create lines for the plots
        self.error_user_line, = self.ax.plot([], [], label='User Error', color='blue')
        self.error_benchmark_line, = self.ax.plot([], [], label='Benchmark', color='red')

        # Add a secondary y-axis for accuracy, associated with ax_difference
        self.ax_accuracy = self.ax_difference.twinx()
        self.accuracy_line, = self.ax_accuracy.plot([], [], label='Accuracy', color='brown')  # Ensure accuracy is added last

        self.ax.legend(loc='upper left')
        self.ax_accuracy.legend(loc='upper right')  # Legend for the accuracy line

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 60)  # Set default FPS to 60

        self.show()

    def update_frame(self):
        if not self.paused:
            try:
                ret_val, image_1 = self.user_cam.read()
                ret_val_1, image_2 = self.benchmark_cam.read()

                if not ret_val or not ret_val_1 or image_1 is None or image_2 is None:
                    print("Error: Couldn't read frames")
                    return

                if self.frame_counter == self.user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.frame_counter = 0
                    self.correct_frames = 0
                    self.user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

                image_1 = self.detector_1.findPose(image_1)
                lmList_user = self.detector_1.findPosition(image_1, draw=False)

                if self.frame_counter == self.benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.frame_counter = 0
                    self.correct_frames = 0
                    self.benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

                image_2 = self.detector_2.findPose(image_2)
                lmList_benchmark = self.detector_2.findPosition(image_2, draw=False)

                self.frame_counter += 1

                error_user, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)
                error_benchmark = 0.0  # Reference error for the benchmark video (set to zero)
                self.errors_user.append(error_user)  # Append error to the list for the user video
                self.errors_benchmark.append(error_benchmark)

                cv2.putText(image_1, 'Error: {}%'.format(str(round(100 * (float(error_user)), 2))), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if error_user < 0.3:
                    cv2.putText(image_1, "CORRECT STEPS", (40, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.correct_frames += 1
                else:
                    cv2.putText(image_1, "INCORRECT STEPS", (40, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - self.fps_time)), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if self.frame_counter == 0:
                    self.frame_counter = self.user_cam.get(cv2.CAP_PROP_FRAME_COUNT)

                Accuracy = round(100 * self.correct_frames / self.frame_counter, 2)
                print(Accuracy)
                self.accuracy.append(Accuracy)
                cv2.putText(image_1, "Dance Steps Accurately Done: {}%".format(
                    str(Accuracy)), (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Resize both videos to have the same height
                common_height = 480  # You can adjust this value
                width_1 = int(image_1.shape[1] * common_height / image_1.shape[0])
                image_1 = cv2.resize(image_1, (width_1, common_height))

                width_2 = int(image_2.shape[1] * common_height / image_2.shape[0])
                image_2 = cv2.resize(image_2, (width_2, common_height))

                # Convert BGR to RGB
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

                concatenated_image = np.concatenate((image_2, image_1 ), axis=1)

                # Update the Matplotlib plots with DTW results
                frames = range(1, len(self.errors_user) + 1)
                self.error_user_line.set_data(frames, self.errors_user)
                self.error_benchmark_line.set_data(frames, self.errors_benchmark)
                self.accuracy_line.set_data(frames, self.accuracy)

                # Relimit and autoscale views for both plots
                self.ax.relim()
                self.ax.autoscale_view()
                self.ax.set_ylim(-2, max(max(self.errors_user), max(self.errors_benchmark)) + 2)

                self.ax_accuracy.relim()  # Relimit for the accuracy axis
                self.ax_accuracy.autoscale_view()  # Autoscale for the accuracy axis
                self.ax_accuracy.set_ylim(0, 100)  # Set the y-axis limit for accuracy

                self.canvas.draw()

                height, width, channel = concatenated_image.shape
                bytesPerLine = 3 * width
                qImg = QImage(concatenated_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.image_label.setPixmap(pixmap)
                self.fps_time = time.time()

            except Exception as e:
                print("Error:", e)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText('Resume')
        else:
            self.pause_button.setText('Pause')

    def update_fps(self, value):
        self.timer.setInterval(1000 // value)

    def closeEvent(self, event):
        self.benchmark_cam.release()
        self.user_cam.release()
        event.accept()


def main():
    benchmark_video = r'Fast Bowling Trimed2.mp4'
    user_video = r'Fast Bowling Trimed.mp4'
    # benchmark_video = r'dance_videos\benchmark_dance.mp4'
    # user_video = r'dance_videos\right_dance.mp4'

    app = QApplication(sys.argv)
    ex = PoseComparisonApp(benchmark_video, user_video)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
