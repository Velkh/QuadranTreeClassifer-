import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QPushButton, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5 import uic


class QuadTreeNode:
    def __init__(self, image, x, y, width, height, threshold):
        self.image = image
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.threshold = threshold
        self.children = []
        self.mean = np.mean(image[y:y + height, x:x + width])

        if width > 1 and height > 1 and np.std(image[y:y + height, x:x + width]) > threshold:
            half_width = width // 2
            half_height = height // 2
            self.children.append(QuadTreeNode(image, x, y, half_width, half_height, threshold))
            self.children.append(QuadTreeNode(image, x + half_width, y, half_width, half_height, threshold))
            self.children.append(QuadTreeNode(image, x, y + half_height, half_width, half_height, threshold))
            self.children.append(
                QuadTreeNode(image, x + half_width, y + half_height, half_width, half_height, threshold))

    def is_leaf(self):
        return len(self.children) == 0


def apply_quad_tree(image, threshold):
    root = QuadTreeNode(image, 0, 0, image.shape[1], image.shape[0], threshold)
    return root


def draw_quad_tree(node, image):
    if node.is_leaf():
        cv2.rectangle(image, (node.x, node.y), (node.x + node.width, node.y + node.height), (255, 0, 0), 1)
    else:
        for child in node.children:
            draw_quad_tree(child, image)


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('showgui.ui', self)

        self.image_label_1 = self.findChild(QLabel, 'image_label_1')
        self.image_label_2 = self.findChild(QLabel, 'image_label_2')
        self.image_label_3 = self.findChild(QLabel, 'image_label_3')
        self.info_text = self.findChild(QTextEdit, 'info_text')
        self.process_button = self.findChild(QPushButton, 'process_button')

        self.actionOpenImage1 = self.findChild(QAction, 'actionOpenImage1')
        self.actionOpenImage2 = self.findChild(QAction, 'actionOpenImage2')
        self.actionOpenImage3 = self.findChild(QAction, 'actionOpenImage3')

        self.actionOpenImage1.triggered.connect(lambda: self.open_image(1))
        self.actionOpenImage2.triggered.connect(lambda: self.open_image(2))
        self.actionOpenImage3.triggered.connect(lambda: self.open_image(3))
        self.process_button.clicked.connect(self.process_images)

        self.image1 = None
        self.image2 = None
        self.image3 = None

    def open_image(self, image_number):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_name:
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if image_number == 1:
                self.image1 = image
                self.display_image(self.image1, self.image_label_1)
            elif image_number == 2:
                self.image2 = image
                self.display_image(self.image2, self.image_label_2)
            elif image_number == 3:
                self.image3 = image
                self.display_image(self.image3, self.image_label_3)

    def display_image(self, image, label):
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:  # rows[0], cols[1], channels[2]
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        label.setPixmap(QPixmap.fromImage(out_image))
        label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        label.setScaledContents(True)

    def process_images(self):
        for image, label in [(self.image1, self.image_label_1), (self.image2, self.image_label_2), (self.image3, self.image_label_3)]:
            if image is not None:
                start_time = time.time()
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                threshold = 10
                quad_tree_root = apply_quad_tree(edges, threshold)
                output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                draw_quad_tree(quad_tree_root, output_image)
                end_time = time.time()

                execution_time = end_time - start_time
                pixel_count = image.size
                unique_colors = len(np.unique(image))

                self.display_image(output_image, label)
                self.info_text.append(f"Image: {label.objectName()}")
                self.info_text.append(f"Resolution: {image.shape[1]} x {image.shape[0]}")
                self.info_text.append(f"Execution Time: {execution_time:.2f} seconds")
                self.info_text.append(f"Pixel Count: {pixel_count}")
                self.info_text.append(f"Unique Colors: {unique_colors}")
                self.info_text.append("=======================================")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
