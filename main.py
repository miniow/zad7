import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QComboBox, QStackedWidget, QTabWidget,
    QAction, QMenu, QToolBar, QStatusBar
)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QKeySequence
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QPoint, QTimer

class Point:
    def __init__(self, x, y):
        self.coords = np.array([x, y, 1])  # Homogeneous coordinates

    def update_coords(self, x, y):
        self.coords = np.array([x, y, 1])

class VectorObject:
    def __init__(self, points=None):
        if points is None:
            points = []
        self.points = points  # List of Point instances

    def add_point(self, point):
        self.points.append(point)

    def remove_point(self, index):
        if 0 <= index < len(self.points):
            del self.points[index]

    def apply_transformation(self, matrix):
        for point in self.points:
            point.coords = matrix @ point.coords

    def get_coordinates(self):
        return [(p.coords[0], p.coords[1]) for p in self.points]

    def copy(self):
        return VectorObject(points=[Point(p.coords[0], p.coords[1]) for p in self.points])

class Canvas(QWidget):
    POINT_RADIUS = 5  # Control point radius
    point_moved = pyqtSignal(int, float, float)
    point_added = pyqtSignal(float, float)
    point_deleted = pyqtSignal(int)
    translation_performed = pyqtSignal(float, float)  # h, v
    rotation_performed = pyqtSignal(float)  # angle in degrees
    scaling_performed = pyqtSignal(float)  # scaling factor k
    rotation_point_defined = pyqtSignal(float, float)
    scaling_point_defined = pyqtSignal(float, float)
    object_transformed = pyqtSignal()

    def __init__(self, vector_object):
        super().__init__()
        self.vector_object = vector_object
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: white;")
        self.selected_point_index = None  # Currently selected point index
        self.dragging = False
        self.operation_mode = "None"
        self.rotation_center = None
        self.scaling_center = None
        self.last_mouse_pos = None
        self.hovered_point_index = None
        self.zoom_scale = 1.0
        self.total_translation = np.array([0.0, 0.0])
        self.total_rotation_angle = 0.0
        self.total_scale_factor = 1.0
        self.pan_offset = np.array([0.0, 0.0])
        self.setMouseTracking(True)

    def set_operation_mode(self, mode):
        self.operation_mode = mode
        self.rotation_center = None
        self.scaling_center = None
        self.total_translation = np.array([0.0, 0.0])
        self.total_rotation_angle = 0.0
        self.total_scale_factor = 1.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0, 0, 0), 2)
        painter.setPen(pen)
        brush = QBrush(QColor(255, 0, 0))
        painter.setBrush(brush)

        # Draw grid and axes
        self.draw_grid(painter)
        self.draw_axes(painter)

        coords = self.vector_object.get_coordinates()
        if not coords:
            return

        # Transform points
        canvas_center = np.array([self.width() / 2, self.height() / 2])
        transformed_coords = [self.transform_point(np.array([x, y])) for x, y in coords]

        # Draw lines between points
        pen = QPen(QColor(0, 0, 255), 2)
        painter.setPen(pen)
        for i in range(len(transformed_coords)):
            start = transformed_coords[i]
            end = transformed_coords[(i + 1) % len(transformed_coords)]  # Close the polygon
            painter.drawLine(QPointF(start[0], start[1]), QPointF(end[0], end[1]))

        # Draw control points
        for index, coord in enumerate(transformed_coords):
            if index == self.hovered_point_index:
                pen = QPen(QColor(255, 215, 0), 2)  # Gold color for hovered point
                brush = QBrush(QColor(255, 215, 0))
            else:
                pen = QPen(QColor(255, 0, 0), 1)
                brush = QBrush(QColor(255, 0, 0))
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(QPointF(coord[0], coord[1]), self.POINT_RADIUS, self.POINT_RADIUS)

        # Draw rotation center if defined
        if self.rotation_center is not None:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 255, 0))
            painter.setBrush(brush)
            center = self.transform_point(self.rotation_center)
            painter.drawEllipse(QPointF(center[0], center[1]), self.POINT_RADIUS + 2, self.POINT_RADIUS + 2)

        # Draw scaling center if defined
        if self.scaling_center is not None:
            pen = QPen(QColor(255, 165, 0), 2)
            painter.setPen(pen)
            brush = QBrush(QColor(255, 165, 0))
            painter.setBrush(brush)
            center = self.transform_point(self.scaling_center)
            painter.drawEllipse(QPointF(center[0], center[1]), self.POINT_RADIUS + 2, self.POINT_RADIUS + 2)

    def draw_axes(self, painter):
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        canvas_center = np.array([self.width() / 2, self.height() / 2])
        # X-axis
        x_start = self.transform_point(np.array([-canvas_center[0], 0]))
        x_end = self.transform_point(np.array([canvas_center[0], 0]))
        painter.drawLine(QPointF(x_start[0], x_start[1]), QPointF(x_end[0], x_end[1]))
        # Y-axis
        y_start = self.transform_point(np.array([0, -canvas_center[1]]))
        y_end = self.transform_point(np.array([0, canvas_center[1]]))
        painter.drawLine(QPointF(y_start[0], y_start[1]), QPointF(y_end[0], y_end[1]))

    def draw_grid(self, painter):
        pen = QPen(QColor(220, 220, 220), 1)
        painter.setPen(pen)
        step = 50 * self.zoom_scale  # Adjust grid spacing based on zoom
        width = self.width()
        height = self.height()
        # Vertical lines
        x = self.pan_offset[0] % step
        while x < width:
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
            x += step
        # Horizontal lines
        y = self.pan_offset[1] % step
        while y < height:
            painter.drawLine(QPointF(0, y), QPointF(width, y))
            y += step

    def transform_point(self, point):
        # Apply zoom and pan transformations
        point = point * self.zoom_scale
        point += self.pan_offset
        point += np.array([self.width() / 2, self.height() / 2])
        point[1] = self.height() - point[1]  # Flip y-axis
        return point

    def inverse_transform_point(self, point):
        # Inverse of transform_point
        point = np.array([point.x(), point.y()], dtype=float)
        point[1] = self.height() - point[1]  # Flip y-axis
        point -= np.array([self.width() / 2, self.height() / 2])
        point -= self.pan_offset
        point /= self.zoom_scale
        return point

    def mousePressEvent(self, event):
        pos = event.pos()
        logical_pos = self.inverse_transform_point(pos)
        x = logical_pos[0]
        y = logical_pos[1]

        if event.button() == Qt.LeftButton:
            if self.operation_mode == "Translation":
                self.dragging = True
                self.last_mouse_pos = np.array([x, y])
                self.total_translation = np.array([0.0, 0.0])
            elif self.operation_mode == "Rotation":
                if self.rotation_center is None:
                    self.rotation_center = np.array([x, y])
                    self.rotation_point_defined.emit(x, y)
                    self.update()
                else:
                    self.dragging = True
                    self.initial_mouse_pos = np.array([x, y])
                    self.last_mouse_pos = self.initial_mouse_pos
                    # Calculate initial angle
                    dx = self.initial_mouse_pos[0] - self.rotation_center[0]
                    dy = self.initial_mouse_pos[1] - self.rotation_center[1]
                    self.initial_angle = np.arctan2(dy, dx)
                    self.last_rotation_angle = 0.0
                    self.total_rotation_angle = 0.0
            elif self.operation_mode == "Scaling":
                if self.scaling_center is None:
                    self.scaling_center = np.array([x, y])
                    self.scaling_point_defined.emit(x, y)
                    self.update()
                else:
                    self.dragging = True
                    self.last_mouse_pos = np.array([x, y])
                    self.total_scale_factor = 1.0
            else:
                index = self.get_point_at_position(pos)
                if index is not None:
                    self.selected_point_index = index
                    self.dragging = True
        elif event.button() == Qt.RightButton:
            index = self.get_point_at_position(pos)
            if index is not None:
                self.vector_object.remove_point(index)
                self.update()
                self.point_deleted.emit(index)
            else:
                new_point = Point(x, y)
                self.vector_object.add_point(new_point)
                self.update()
                self.point_added.emit(x, y)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        logical_pos = self.inverse_transform_point(pos)
        x = logical_pos[0]
        y = logical_pos[1]
        current_mouse_pos = np.array([x, y])

        # Hover effect
        index = self.get_point_at_position(event.pos())
        if index != self.hovered_point_index:
            self.hovered_point_index = index
            self.update()

        if self.dragging:
            if self.operation_mode == "Translation":
                delta = current_mouse_pos - self.last_mouse_pos
                translation_matrix = np.array([
                    [1, 0, delta[0]],
                    [0, 1, delta[1]],
                    [0, 0, 1]
                ])
                self.vector_object.apply_transformation(translation_matrix)
                self.total_translation += delta
                self.last_mouse_pos = current_mouse_pos
                self.update()
                self.object_transformed.emit()
                self.translation_performed.emit(self.total_translation[0], self.total_translation[1])
            elif self.operation_mode == "Rotation" and self.rotation_center is not None:
                dx = current_mouse_pos[0] - self.rotation_center[0]
                dy = current_mouse_pos[1] - self.rotation_center[1]
                current_angle = np.arctan2(dy, dx)
                # Unwrap the angle to handle wrapping
                angle = np.unwrap([self.initial_angle, current_angle])[1] - self.initial_angle
                angle_deg = np.degrees(angle)
                # Calculate the incremental angle to apply
                delta_angle_deg = angle_deg - self.last_rotation_angle
                rotation_matrix = self.get_rotation_matrix(
                    self.rotation_center[0], self.rotation_center[1], delta_angle_deg)
                self.vector_object.apply_transformation(rotation_matrix)
                self.last_rotation_angle = angle_deg
                self.total_rotation_angle = angle_deg
                self.update()
                self.object_transformed.emit()
                self.rotation_performed.emit(self.total_rotation_angle)
            elif self.operation_mode == "Scaling" and self.scaling_center is not None:
                v1 = self.last_mouse_pos - self.scaling_center
                v2 = current_mouse_pos - self.scaling_center
                if np.linalg.norm(v1) == 0:
                    scale_factor = 1
                else:
                    scale_factor = np.linalg.norm(v2) / np.linalg.norm(v1)
                scaling_matrix = self.get_scaling_matrix(self.scaling_center[0], self.scaling_center[1], scale_factor)
                self.vector_object.apply_transformation(scaling_matrix)
                self.total_scale_factor *= scale_factor
                self.last_mouse_pos = current_mouse_pos
                self.update()
                self.object_transformed.emit()
                self.scaling_performed.emit(self.total_scale_factor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.selected_point_index = None
            if self.operation_mode == "Translation":
                self.translation_performed.emit(0.0, 0.0)
            elif self.operation_mode == "Rotation":
                self.rotation_performed.emit(0.0)
            elif self.operation_mode == "Scaling":
                self.scaling_performed.emit(1.0)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.zoom_scale *= factor
        self.update()

    def get_point_at_position(self, pos):
        """Returns the index of the point if the cursor is near it, otherwise None."""
        coords = self.vector_object.get_coordinates()
        for index, (x, y) in enumerate(coords):
            transformed_point = self.transform_point(np.array([x, y]))
            distance = np.hypot(pos.x() - transformed_point[0], pos.y() - transformed_point[1])
            if distance <= self.POINT_RADIUS + 2:  # Small margin
                return index
        return None

    def get_rotation_matrix(self, x0, y0, alpha_deg):
        alpha = np.radians(alpha_deg)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        # Translation to origin
        to_origin = np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0, 1]
        ])
        # Rotation matrix
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        # Translation back
        back = np.array([
            [1, 0, x0],
            [0, 1, y0],
            [0, 0, 1]
        ])
        return back @ rotation @ to_origin

    def get_scaling_matrix(self, xs, ys, k):
        scaling_matrix = np.array([
            [k, 0, xs * (1 - k)],
            [0, k, ys * (1 - k)],
            [0, 0, 1]
        ])
        return scaling_matrix

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Vector Object Transformation")
        self.vector_object = VectorObject()
        self.undo_stack = []
        self.redo_stack = []
        self.init_ui()

    def points_table_cell_changed(self, row, column):
        try:
            x_item = self.points_table.item(row, 0)
            y_item = self.points_table.item(row, 1)
            x = float(x_item.text())
            y = float(y_item.text())
            self.vector_object.points[row].update_coords(x, y)
            self.canvas.update()
            self.save_state()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid x and y values.")
    def update_translation_inputs(self, h, v):
        if self.operation_combo.currentText() == "Translation":
            self.h_input.blockSignals(True)
            self.v_input.blockSignals(True)
            self.h_input.setText(f"{h:.2f}")
            self.v_input.setText(f"{v:.2f}")
            self.h_input.blockSignals(False)
            self.v_input.blockSignals(False)

    def update_rotation_input(self, angle_deg):
        if self.operation_combo.currentText() == "Rotation":
            self.alpha_input.blockSignals(True)
            self.alpha_input.setText(f"{angle_deg:.2f}")
            self.alpha_input.blockSignals(False)

    def update_scaling_input(self, k):
        if self.operation_combo.currentText() == "Scaling":
            self.k_input.blockSignals(True)
            self.k_input.setText(f"{k:.2f}")
            self.k_input.blockSignals(False)

    def init_ui(self):
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        canvas_layout = QVBoxLayout()

        # Canvas
        self.canvas = Canvas(self.vector_object)
        self.canvas.object_transformed.connect(self.update_points_table)
        self.canvas.translation_performed.connect(self.update_translation_inputs)
        self.canvas.rotation_performed.connect(self.update_rotation_input)
        self.canvas.scaling_performed.connect(self.update_scaling_input)
        canvas_layout.addWidget(self.canvas)

        # Controls
        self.tabs = QTabWidget()

        # Add Point Tab
        add_point_tab = QWidget()
        add_point_layout = QVBoxLayout()
        add_point_label = QLabel("Add Point (x, y):")
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("x")
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("y")
        add_point_button = QPushButton("Add Point")
        add_point_button.clicked.connect(self.add_point)
        add_point_layout.addWidget(add_point_label)
        add_point_layout.addWidget(self.x_input)
        add_point_layout.addWidget(self.y_input)
        add_point_layout.addWidget(add_point_button)
        add_point_tab.setLayout(add_point_layout)

        # Transformation Tab
        transform_tab = QWidget()
        transform_layout = QVBoxLayout()

        # Operation Selection
        operation_label = QLabel("Select Operation:")
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["None", "Translation", "Rotation", "Scaling"])
        self.operation_combo.currentIndexChanged.connect(self.operation_changed)

        # Stacked Widget for operation controls
        self.operation_stack = QStackedWidget()

        # None Operation (Placeholder)
        none_widget = QWidget()
        none_layout = QVBoxLayout()
        none_layout.addWidget(QLabel("Select an operation to perform."))
        none_widget.setLayout(none_layout)

        # Translation Controls
        translation_widget = QWidget()
        translation_layout = QVBoxLayout()
        translation_label = QLabel("Translation (h, v):")
        self.h_input = QLineEdit()
        self.h_input.setPlaceholderText("h")
        self.v_input = QLineEdit()
        self.v_input.setPlaceholderText("v")
        translate_button = QPushButton("Translate")
        translate_button.clicked.connect(self.translate)
        translation_layout.addWidget(translation_label)
        translation_layout.addWidget(self.h_input)
        translation_layout.addWidget(self.v_input)
        translation_layout.addWidget(translate_button)
        translation_widget.setLayout(translation_layout)

        # Rotation Controls
        rotation_widget = QWidget()
        rotation_layout = QVBoxLayout()
        rotation_label = QLabel("Rotation:")
        self.x0_input = QLineEdit()
        self.x0_input.setPlaceholderText("x0")
        self.y0_input = QLineEdit()
        self.y0_input.setPlaceholderText("y0")
        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText("α (degrees)")
        rotate_button = QPushButton("Rotate")
        rotate_button.clicked.connect(self.rotate)
        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.x0_input)
        rotation_layout.addWidget(self.y0_input)
        rotation_layout.addWidget(self.alpha_input)
        rotation_layout.addWidget(rotate_button)
        rotation_widget.setLayout(rotation_layout)

        # Scaling Controls
        scaling_widget = QWidget()
        scaling_layout = QVBoxLayout()
        scaling_label = QLabel("Scaling:")
        self.xs_input = QLineEdit()
        self.xs_input.setPlaceholderText("xs")
        self.ys_input = QLineEdit()
        self.ys_input.setPlaceholderText("ys")
        self.k_input = QLineEdit()
        self.k_input.setPlaceholderText("k")
        scale_button = QPushButton("Scale")
        scale_button.clicked.connect(self.scale)
        scaling_layout.addWidget(scaling_label)
        scaling_layout.addWidget(self.xs_input)
        scaling_layout.addWidget(self.ys_input)
        scaling_layout.addWidget(self.k_input)
        scaling_layout.addWidget(scale_button)
        scaling_widget.setLayout(scaling_layout)

        # Add operation widgets to stack
        self.operation_stack.addWidget(none_widget)
        self.operation_stack.addWidget(translation_widget)
        self.operation_stack.addWidget(rotation_widget)
        self.operation_stack.addWidget(scaling_widget)

        transform_layout.addWidget(operation_label)
        transform_layout.addWidget(self.operation_combo)
        transform_layout.addWidget(self.operation_stack)
        transform_tab.setLayout(transform_layout)

        # Points Tab
        points_tab = QWidget()
        points_layout = QVBoxLayout()
        points_label = QLabel("Point List:")
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(2)
        self.points_table.setHorizontalHeaderLabels(["x", "y"])
        self.points_table.horizontalHeader().setStretchLastSection(True)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.points_table.cellChanged.connect(self.points_table_cell_changed)
        delete_point_button = QPushButton("Delete Selected Point")
        delete_point_button.clicked.connect(self.delete_selected_point)
        points_layout.addWidget(points_label)
        points_layout.addWidget(self.points_table)
        points_layout.addWidget(delete_point_button)
        points_tab.setLayout(points_layout)

        # Add tabs to tab widget
        self.tabs.addTab(add_point_tab, "Add Point")
        self.tabs.addTab(transform_tab, "Transform")
        self.tabs.addTab(points_tab, "Points")

        # ToolBar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_object)
        toolbar.addAction(reset_action)

        load_action = QAction("Load", self)
        load_action.triggered.connect(self.load_object)
        toolbar.addAction(load_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_object)
        toolbar.addAction(save_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add widgets to control layout
        control_layout.addWidget(self.tabs)

        self.canvas.point_moved.connect(self.canvas_point_moved)
        self.canvas.point_added.connect(self.canvas_point_added)
        self.canvas.point_deleted.connect(self.canvas_point_deleted)
        self.canvas.rotation_point_defined.connect(self.rotation_point_defined)
        self.canvas.scaling_point_defined.connect(self.scaling_point_defined)

        # Set layouts
        main_layout.addLayout(control_layout)
        main_layout.addLayout(canvas_layout)
        main_widget.setLayout(main_layout)

        self.save_state()

    def operation_changed(self, index):
        operation = self.operation_combo.currentText()
        self.canvas.set_operation_mode(operation)
        self.operation_stack.setCurrentIndex(index)
        if operation != "Rotation":
            self.canvas.rotation_center = None
        if operation != "Scaling":
            self.canvas.scaling_center = None
        self.canvas.update()
        self.status_bar.showMessage(f"Current Operation: {operation}")

    def rotation_point_defined(self, x, y):
        self.x0_input.setText(f"{x:.2f}")
        self.y0_input.setText(f"{y:.2f}")
        self.status_bar.showMessage(f"Rotation center set at ({x:.2f}, {y:.2f})")

    def scaling_point_defined(self, x, y):
        self.xs_input.setText(f"{x:.2f}")
        self.ys_input.setText(f"{y:.2f}")
        self.status_bar.showMessage(f"Scaling center set at ({x:.2f}, {y:.2f})")

    def canvas_point_moved(self, index, x, y):
        self.points_table.blockSignals(True)
        self.points_table.setItem(index, 0, QTableWidgetItem(f"{x:.2f}"))
        self.points_table.setItem(index, 1, QTableWidgetItem(f"{y:.2f}"))
        self.points_table.blockSignals(False)
        self.save_state()

    def canvas_point_added(self, x, y):
        self.update_points_table()
        self.save_state()

    def canvas_point_deleted(self, index):
        self.update_points_table()
        self.save_state()

    def add_point(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            point = Point(x, y)
            self.vector_object.add_point(point)
            self.x_input.clear()
            self.y_input.clear()
            self.canvas.update()
            self.update_points_table()  # Update the table
            self.save_state()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid x and y values.")

    def update_points_table(self):
        self.points_table.blockSignals(True)  # Block signals to avoid recursion
        self.points_table.setRowCount(len(self.vector_object.points))
        for i, point in enumerate(self.vector_object.points):
            x_item = QTableWidgetItem(f"{point.coords[0]:.2f}")
            y_item = QTableWidgetItem(f"{point.coords[1]:.2f}")
            self.points_table.setItem(i, 0, x_item)
            self.points_table.setItem(i, 1, y_item)
        self.points_table.blockSignals(False)

    def delete_selected_point(self):
        selected_rows = self.points_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            self.vector_object.remove_point(row)
            self.update_points_table()
            self.canvas.update()
            self.save_state()

    def translate(self):
        try:
            h = float(self.h_input.text())
            v = float(self.v_input.text())
            translation_matrix = np.array([
                [1, 0, h],
                [0, 1, v],
                [0, 0, 1]
            ])
            self.vector_object.apply_transformation(translation_matrix)
            self.h_input.clear()
            self.v_input.clear()
            self.canvas.update()
            self.update_points_table()
            self.save_state()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid h and v values.")

    def rotate(self):
        try:
            x0 = float(self.x0_input.text())
            y0 = float(self.y0_input.text())
            alpha_deg = float(self.alpha_input.text())
            transformation_matrix = self.canvas.get_rotation_matrix(x0, y0, alpha_deg)
            self.vector_object.apply_transformation(transformation_matrix)
            self.alpha_input.clear()
            self.canvas.update()
            self.update_points_table()
            self.save_state()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid rotation values.")

    def scale(self):
        try:
            xs = float(self.xs_input.text())
            ys = float(self.ys_input.text())
            k = float(self.k_input.text())
            scaling_matrix = self.canvas.get_scaling_matrix(xs, ys, k)
            self.vector_object.apply_transformation(scaling_matrix)
            self.k_input.clear()
            self.canvas.update()
            self.update_points_table()
            self.save_state()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid scaling values.")

    def reset_object(self):
        reply = QMessageBox.question(self, 'Reset Object', 'Are you sure you want to reset the object?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.vector_object = VectorObject()
            self.canvas.vector_object = self.vector_object
            self.canvas.update()
            self.update_points_table()
            self.save_state()

    def load_object(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Object", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    lines = f.readlines()
                points = []
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        continue
                    x, y = map(float, parts)
                    points.append(Point(x, y))
                self.vector_object = VectorObject(points)
                self.canvas.vector_object = self.vector_object
                self.canvas.update()
                self.update_points_table()
                self.save_state()
                self.status_bar.showMessage(f"Loaded object from {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot load file: {e}")

    def save_object(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Object", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    for point in self.vector_object.points:
                        f.write(f"{point.coords[0]},{point.coords[1]}\n")
                QMessageBox.information(self, "Success", "Object saved successfully.")
                self.status_bar.showMessage(f"Saved object to {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot save file: {e}")

    def save_state(self):
        # Save the current state for undo functionality
        self.undo_stack.append(self.vector_object.copy())
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.vector_object = self.undo_stack[-1].copy()
            self.canvas.vector_object = self.vector_object
            self.canvas.update()
            self.update_points_table()
            self.status_bar.showMessage("Undo performed")
        else:
            self.status_bar.showMessage("Nothing to undo")

    def redo(self):
        if self.redo_stack:
            self.vector_object = self.redo_stack[-1].copy()
            self.undo_stack.append(self.redo_stack.pop())
            self.canvas.vector_object = self.vector_object
            self.canvas.update()
            self.update_points_table()
            self.status_bar.showMessage("Redo performed")
        else:
            self.status_bar.showMessage("Nothing to redo")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
