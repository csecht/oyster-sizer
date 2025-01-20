"""
oystersize.py provides size metrics of triploid Crassostrea
virginica oysters from images of population samples for all stages of
aquaculture production. With the user interacting through a tkinter GUI,
the program detects and classifies oysters and size standard objects in
an image. Once the size standard's known radius is entered, oyster sizes,
mean, median, and range are reported, along with an annotated image.
Results can be saved to file. Detection is based on a YOLOv8n model from
Ultralytics using transfer learning. The oyster sizing model was trained
on two custom classes: oyster images at various stages of growth, and
real and synthetic disk-like images as size standards.

Quit the program with Esc key, Ctrl-Q key, the close window icon of the
report window or File menubar. From the command line, use Ctrl-C.
See "Requirements" and "Usage" in the README.md file for more information.

Developed using Python 3.9 through 3.12.7.
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License
# No warranty. Use at your own risk.

# Standard library imports.
from pathlib import Path
from signal import signal, SIGINT
from statistics import mean, median
from sys import exit as sys_exit
from time import time
from typing import Union, List, Tuple

# Third party imports.
# tkinter(Tk/Tcl) is included with most Python3 distributions,
# but may sometimes need to be regarded as third-party.
try:
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, Event
    from torch import device, cuda
    from torch.backends import mps  # used for macOS M1+ chips
    from ultralytics import YOLO
    from ultralytics.utils.ops import xywh2xyxy

except (ImportError, ModuleNotFoundError) as import_err:
    sys_exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\n'
        'OpenCV-Python, NumPy, tkinter (Tk/Tcl), PyTorch, Ultralytics.\n'
        'To install: from the current folder, run this command'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install -r requirements.txt\n'
        'Alternative command formats (system dependent):\n'
        '   py -m pip install -r requirements.txt (Windows)\n'
        '   pip install -r requirements.txt\n'
        'You may also install directly using, for example, this command,'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install ultralytics\n'
        'On Linux, if tkinter is the problem, then you may need:\n'
        '   sudo apt-get install python3-tk\n'
        'See also: https://numpy.org/install/\n'
        '  https://tkdocs.com/tutorial/install.html\n'
        '  https://docs.opencv2.org/4.6.0/d5/de5/tutorial_py_setup_in_windows.html\n'
        'Consider running this app and installing missing packages in a virtual environment.\n'
        f'Error message:\n{import_err}')

# Local application imports.
# To ensure exit messaging, place local imports after try...except imports.
from utility_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             to_precision as to_p)

MY_OS = const.MY_OS
PROGRAM_NAME = utils.program_name()


class ProcessImage(tk.Tk):
    """
    Apply YOLO object detection for tkinter and OpenCV image processing.
    """

    def __init__(self):
        super().__init__()

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  as Label images displayed in their respective tkimg_window Toplevel.
        #  The cvimg images are numpy arrays.
        self.tkimg: dict = {}
        self.cvimg: dict = {}
        for _name in const.WINDOW_TITLES:
            self.tkimg[_name] = tk.PhotoImage()
            self.cvimg[_name] = const.STUB_ARRAY

        # Note: The matching selector widgets for the following
        #  control variables are in ViewImage __init__.
        # Dictionary attribute allows for easy addition of new sliders.
        self.confidence_slide_val = tk.IntVar()

        self.predicted_boxes= np.array([])  # x-ctr, y-ctr, width, height
        self.predicted_class_distribution = np.array([])  # 0 is 'disk', 1 is 'oyster'

    def prediction(self):
        """
        YOLO prediction for oyster input image, with size standard object.
        """

        # Initialize model
        model_to_use = utils.valid_path_to(f"models/{const.MODEL_NAME}/weights/best.pt")
        model = YOLO(model_to_use)

        # Run inference, on CPU if no GPU available.
        available_device = ('mps' if MY_OS == 'dar' and mps.is_available()
                            else device('cuda' if cuda.is_available() else 'cpu')
                            )

        confidence: float = self.confidence_slide_val.get() / 100

        # self.cvimg['input'] defined in open_input() from self.input_file_path
        #  via cv2.imread() of filedialog.askopenfilename().
        results = model.predict(
            source=self.cvimg['input'].copy(),
            imgsz=const.PREDICT_IMGSZ,
            conf=confidence,
            device=available_device,
            iou=const.PREDICT_IOU,
            max_det=const.PREDICT_MAX_DET,
            half=const.PREDICT_HALF,
            verbose=False,
        )

        # Device-agnostic conversion of tensors to numpy arrays.
        box_data = results[0].boxes
        self.predicted_boxes = box_data.xywh.numpy(force=True).astype(int)
        self.predicted_class_distribution = box_data.cls.numpy(force=True).astype(int)


class ViewImage(ProcessImage):
    """
    A suite of methods to display images, YOLO model object detections,
    and results text.
    Methods:
    set_auto_scale_factor
    show_info_message
    widget_control
    update_image
    is_interior
    find_interior_objects
    find_true_pos_objects
    get_standard_sizes
    validate_size_entry
    determine_mean_standard_size
    get_sig_fig
    convert_bbox_data
    get_text_position_offsets
    annotate_object
    display_all_objects
    display_metrics_in_image
    report_results
    display_processing_info
    process_sizes
    process_prediction
    """

    def __init__(self):
        super().__init__()

        self.first_run: bool = True

        self.report_frame = tk.Frame()
        self.selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        self.entry = {
            'size_entry': tk.Entry(master=self.selectors_frame),
            'size_std_val': tk.StringVar(master=self.selectors_frame),
            'size_std_lbl': tk.Label(master=self.selectors_frame),
            'size_std_lbl2': tk.Label(master=self.selectors_frame),
        }

        self.button = {
            'update': ttk.Button(master=self),
            'save_results': ttk.Button(master=self),
            'new_input': ttk.Button(master=self),
        }

        self.slider = {
            'confidence': tk.Scale(master=self.selectors_frame),
            'confidence_lbl': tk.Label(master=self.selectors_frame),
        }


        # img_label dictionary is set up in SetupApp.setup_image_windows(),
        #  but is used in Class methods here.
        self.img_label: dict = {}

        # metrics dict is populated in SetupApp.open_input().
        self.metrics: dict = {}
        self.line_thickness: int = 0
        self.font_scale: float = 0
        self.time_start: float = 0
        self.elapsed: Union[float, int, str] = 0
        self.screen_width: int = 0
        self.scale_factor = tk.DoubleVar()

        self.color_val = tk.StringVar()

        # Info label is gridded in configure_main_window().
        self.info_txt = tk.StringVar()
        self.info_label = tk.Label(textvariable=self.info_txt)

        # Defined in widget_control() to reset values that user may have
        #  tried to change during prolonged processing times.
        self.slider_val_saved: str = ''

        # The following group of attributes is set in SetupApp.open_input().
        self.input_file_path: str = ''
        self.input_file_name: str = ''
        self.input_folder_name: str = ''
        self.input_ht: int = 0
        self.input_w: int = 0

        # Attributes used for filtering, sizing, and reporting.
        self.interior_standards = np.array([])
        self.interior_oysters = np.array([])
        self.true_pos_standards = np.array([])
        self.true_pos_oysters = np.array([])
        self.standards_mean_px_size: float = 0
        self.unit_per_px: float = 0
        self.standards_mean_measured_size = tk.DoubleVar()
        self.bbox_ratio_mean: float = 0
        self.oyster_sizes: List[float] = []
        self.report_txt: str = ''

    def set_auto_scale_factor(self) -> None:
        """
        As a convenience for user, set a default scale factor that will
        easily fit images on the screen; either 1/3 screen px width or
        2/3 screen px height, depending on input image orientation.
        Called from open_input() and call_cmd().apply_default_settings().

        Returns: None
        """

        if self.input_w >= self.input_ht:
            estimated_scale = round((self.screen_width * 0.33) / self.input_w, 2)
        else:
            estimated_scale = round((self.winfo_screenheight() * 0.66) / self.input_ht, 2)

        self.scale_factor.set(estimated_scale)

    def show_info_message(self, info: str, color: str) -> None:
        """
        Configure for display and update the informational message in
        the report and settings window.
        Args:
            info: The text string of the message to display.
            color: The font color string, either as a key in the
                   const.COLORS_TK dictionary or as a Tk compatible fg
                   color string, i.e. hex code or X11 named color.

        Returns: None
        """
        self.info_txt.set(info)
        self.info_label.config(fg=const.COLORS_TK.get(color, color))

    def widget_control(self, action: str) -> None:
        """
        Used to disable settings widgets when processing is running.
        Provides a watch cursor while widgets are disabled. Also, gets
        Scale() values at time of disabling and resets them upon
        enabling, thus preventing user click events, which are retained
        in memory during processing, from changing slider position
        post-processing. Called from process_prediction().
        Args:
            action: Either 'off' to disable widgets, or 'on' to enable.
        Returns:
            None
        """

        if action == 'off':
            self.slider['confidence'].configure(state=tk.DISABLED)

            # Grab the current slider values, in case user tries to change.
            self.slider_val_saved = self.confidence_slide_val.get()
            for _, _w in self.button.items():
                _w.grid_remove()
                self.show_info_message(info='\nProcessing...\n\n', color='black')
            for _, _w in self.entry.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.DISABLED)
            self.config(cursor='watch')
        else:  # is 'on'
            self.slider['confidence'].configure(state=tk.NORMAL)

            # Restore the slider values to overwrite any changes.
            self.confidence_slide_val.set(self.slider_val_saved)
            for _, _w in self.button.items():
                _w.grid()
            for _, _w in self.entry.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.NORMAL)
            self.config(cursor='')
            self.slider_val_saved = ''

        # Use update(), not update_idletasks, here to speed up windows'
        #  response.
        self.update()

    def update_image(self, img_name: str) -> None:
        """
        Process a cv2 image array to use as a tk PhotoImage and update
        (configure) its window label for immediate display, at scale.
        Calls module manage.tk_image(). Called from all methods that
        display an image.

        Args:
            img_name: An item name used in the image_name tuple, for
                use as key in tkimg, cvimg, and img_label dictionaries.

        Returns:
            None
        """

        self.tkimg[img_name] = manage.tk_image(
            image=self.cvimg[img_name],
            scale_factor=self.scale_factor.get()
        )

        self.img_label[img_name].configure(image=self.tkimg[img_name])

    def is_interior(self, xywh_bbox) -> bool:
        """
        Filter an object's bouncing box position within a few pixels
        of the input image border.
        Uses ultralytics.utils.ops.xywh2xyxy() to convert YOLO format
        to cv2 rectangle format x1, y1, x2, y2.
        Called from find_interior_objects().

        Args:
            xywh_bbox: A numpy array of YOLO bounding box coordinates;
            x_center, y_center, width, height

        Returns:
            True if the box is completely within the input image,
            False if not.
        """

        # x_center, y_center, width, height = xywh_bbox
        # x1, y1 = x_center - width / 2, y_center - height / 2
        # x2, y2 = x_center + width / 2, y_center + height / 2
        x1, y1, x2, y2 = xywh2xyxy(xywh_bbox)

        # Set limits for coordinate points to identify boxes that
        # are within a few pixels of an image file border (edge).
        return not (x1 <= const.EDGE_PROXIMITY
                    or y1 <= const.EDGE_PROXIMITY
                    or x2 >= self.input_w - const.EDGE_PROXIMITY
                    or y2 >= self.input_ht - const.EDGE_PROXIMITY)

    def find_interior_objects(self) -> None:
        """
        Filter out objects that are on or near the image border.
        Popup window warns user if no objects are found.
        Calls is_interior(), utils.no_objects_found_msg().
        Called from process_sizes().
        Returns: None
        """

        # NOTE: predicted_boxes and predicted_class_distribution are
        # defined in prediction(), which is called from process_prediction().
        # In predicted_class_distribution, 0 is 'disk' and 1 is 'oyster' training class names.
        std_objects: np.ndarray = self.predicted_boxes[self.predicted_class_distribution == 0]
        oyster_objects: np.ndarray = self.predicted_boxes[self.predicted_class_distribution == 1]

        self.interior_standards = (
            std_objects[np.apply_along_axis(func1d=self.is_interior, axis=1, arr=std_objects)]
            if std_objects.size else np.array([])
        )
        self.interior_oysters = (
            oyster_objects[np.apply_along_axis(func1d=self.is_interior, axis=1, arr=oyster_objects)]
            if oyster_objects.size else np.array([])
        )

        if not std_objects.size:
            utils.no_objects_found_msg(caller='std_objects')
        elif not self.interior_standards.size:
            utils.no_objects_found_msg(caller='interior_std')

        if not oyster_objects.size:
            utils.no_objects_found_msg(caller='oyster_objects')
        elif not self.interior_oysters.size:
            utils.no_objects_found_msg(caller='interior_oyster')

    def find_true_pos_objects(self) -> None:
        """
        Filter out once class of objects that are also detected as other
        class, that is, find true positive class detections.
        Defines true_pos_standards and true_pos_oysters.
        Need to call after find_interior_objects() and before
        determine_mean_standard_size(). Called from process_sizes().
        Calls utils.box_is_very_close_inarray().
        Returns: None
        """
        # Note: may need to adjust closeness threshold tolerance of
        #   utils.centered_boxes_very_close(); 0.1 is a good starting point.
        are_not_close_to_standards = np.array(
            [not utils.box_is_very_close_inarray(box, self.interior_standards, 0.1)
             for box in self.interior_oysters]
        )

        are_not_close_to_oysters = np.array(
            [not utils.box_is_very_close_inarray(box, self.interior_oysters,0.1)
             for box in self.interior_standards]
        )

        # This vectorized approach may not be faster than using list expressions,
        #  but, when properly named, it is more readable and maintainable.
        #  And it is a good example of using numpy array indexing.
        self.true_pos_standards = (
            self.interior_standards[are_not_close_to_oysters]
            if are_not_close_to_oysters.size else np.array([])
        )

        self.true_pos_oysters = (
            self.interior_oysters[are_not_close_to_standards]
            if are_not_close_to_standards.size else np.array([])
        )

    def set_bbox_ratio_mean(self, bbox_ary) -> None:
        """
        Calculate the width and height bounding box ratios of objects
        and set the mean for setting a size correction factor. Generally
        used for oysters, but can be used for standards as well.
        Called from process_sizes().

        Args:
            bbox_ary: A numpy array of bounding boxes in xywh format.
        Returns: None
        """
        if bbox_ary.size:
            wh_max = bbox_ary[:, 2:].max(axis=1)
            wh_min = bbox_ary[:, 2:].min(axis=1)
            self.bbox_ratio_mean = (wh_max / wh_min).mean()
        else:
            self.bbox_ratio_mean = 0

    def get_standard_sizes(self) -> np.ndarray:
        """
        Return the maximum of pixel widths and heights of valid size
        standards from a ndarray of one or more xywh bounding boxes.
        Need to confirm truth of true_pos_standards.size before calling.
        Called from determine_mean_standard_size(), display_processing_info().

        Returns: A numpy array of standards' maximum pixel dimension.
        """
        return self.true_pos_standards[:, 2:].max(axis=1)

    def validate_size_entry(self) -> None:
        """
        Check whether custom size Entry() value is a real number.
        Corrects the Entry() value if it is not or posts a message if
        entry is otherwise not valid.
        Called from process_sizes().
        Returns: None
        """
        # Verify that entries are positive numbers.
        # Custom sizes can be entered only as integer or float.
        entered_num = ''.join(_c for _c in self.entry['size_std_val'].get()
                          if (_c.isdigit() or _c in {'.', '_'})
                          )
        try:
            # float() will raise ValueError if entry is not valid.
            if not float(entered_num):
                raise ValueError
            self.entry['size_std_val'].set(entered_num)
        except ValueError:
            messagebox.showinfo(title='Custom size',
                                detail='Enter a number > 0.\n'
                                       'Accepted types:\n'
                                       '  integer: 26, 2651, 2_651 or 2,651\n'
                                       '  decimal: 26.5, 0.265, .2\n'
                                )
            self.entry['size_std_val'].set('1')


    def determine_mean_standard_size(self) -> None:
        """
        Calculate the mean size of the standard the custom size standard
        value in the Entry widget. Calculate the unit_per_px value.
        If no size standard found, mean size reported as 'n/a'.
        Called from process_sizes().

        Returns: None
        """

        # Note: no standard objects found warn msg is in find_interior_objects().
        if not self.true_pos_standards.size:
            self.entry['size_std_val'].set('1')
            return

        # Flag from display_processing_info() if standards' sizes are non-concordant.
        # Note: keep mean as string for proper SF evaluation in get_sig_fig().
        std_sizes = self.get_standard_sizes()
        self.standards_mean_px_size: str = to_p.to_precision(
            value=std_sizes.mean(),
            precision=utils.count_sig_fig(std_sizes.min())
        )

        # Get the entered standard size value and calculate the mean size.
        # Note: standards_mean_measured_size is used only for reporting and
        #  is converted to correct sig. fig. in report_results().
        #  unit_per_px is used in convert_bbox_data() to calculate
        #  object sizes.
        size_std_val = float(self.entry['size_std_val'].get())
        self.unit_per_px = size_std_val / float(self.standards_mean_px_size)
        std_calc_sizes: np.ndarray = std_sizes * self.unit_per_px
        self.standards_mean_measured_size.set(std_calc_sizes.mean())

    def get_sig_fig(self) -> int:
        """
        Calculate the number of significant figures to display based on
        the lesser of the entered custom size standard value or the
        standard mean pixel diameter. Called from convert_bbox_data(),
        display_metrics_in_image(), report_results(), process_sizes().
        Calls utils.count_sig_fig().

        Returns: None
        """
        # Note: standards_mean_px_size is defined in determine_mean_standard_size().
        return min(utils.count_sig_fig(self.entry['size_std_val'].get()),
                   utils.count_sig_fig(self.standards_mean_px_size))

    def convert_bbox_data(self, bbox: np.ndarray) -> tuple:
        """
        Convert bounding box xywh to a xyxy format for drawing a cv2
        rectangle; measure object's size from its longest box dimension
        multiplied by the unit_per_px factor.
        Called from display_all_objects().
        Calls to_p.to_precision() to set the number of significant figures.

        Args:
             bbox: A numpy array for a single bounding box element, in
                   xywh centered-box format.
        Returns: tuple of data converted to use in cv2.rectangle, as
            (x1, y1, x2, y2), and the object length, as text.
        """

        x1, y1, x2, y2 = xywh2xyxy(bbox)
        longest_box_side = bbox[2:].max()  # max of width and height, pixels.
        calculated_size: float = longest_box_side * self.unit_per_px

        # Need to apply sig. fig. for sizes in annotated image and report.
        display_size: str = to_p.to_precision(value=calculated_size,
                                              precision=self.get_sig_fig())

        if self.entry['size_std_val'].get() == '1':
            display_size = f'{longest_box_side}px'

        return x1, y1, x2, y2, display_size

    def get_text_position_offsets(self, txt_string: str) -> Tuple[float, int]:
        """
        Calculate the x and y position correction factors to help center
        *txt_string* in cv2.putText() for annotating objects.
        Called from annotate_object().

        Args:
            txt_string: A string of the object's size to display.
        Returns:
            A tuple of x and y position adjustment factors for size
            annotation.
        """

        ((txt_width, _), baseline) = cv2.getTextSize(
            text=txt_string,
            fontFace=const.FONT_TYPE,
            fontScale=self.font_scale,
            thickness=self.line_thickness)
        offset_x = txt_width / 2

        return offset_x, baseline

    def annotate_object(self,
                        point1: tuple,
                        point2: tuple,
                        object_size: str,
                        object_name: str) -> None:
        """
        Draw a rectangle around the object and annotate its size.
        Called from display_all_objects().
        Calls get_text_position_offsets
        Args:
            point1: An x,y tuple of the top-left corner of the bounding box.
            point2: An x,y tuple of the bottom-right corner of the bounding box.
            object_size: A string of the object's size to display.
            object_name: A string of the predicted object, either
                'standard' for size standard or 'oyster' for oyster; used
                to specify the annotation color.
        Returns: None
        """

        color_selection: tuple = const.COLORS_CV.get(self.color_val.get(), 'green')
        if object_name == 'standard':
            color_selection = const.COLORS_CV['DarkOrchid1']

        # Use letter 'O' for oyster annotation instead of its measured size.
        # if object_name == 'oyster':
        #     object_size = 'O'

        # Draw the bounding box rectangle around the object.
        arr = cv2.rectangle(img=self.cvimg['sized'],
                            pt1=point1,
                            pt2=point2,
                            color=color_selection,
                            thickness=self.line_thickness,
                            )

        # Use the given center percentage of the box area to determine
        #  the best text color contrast in the annotation area.
        # Keep in mind that the center rect of small objects may be smaller
        #  than the annotation text, so text contrast may not be optimal.
        #  Adjusting the center_pct value may help.
        box = arr[point1[1]:point2[1], point1[0]:point2[0], :]
        text_contrast = utils.auto_text_contrast(box_area=box, center_pct=0.3)

        # Center the size text in the bounding box rectangle with org param.
        #  org: bottom-left corner of the text annotation for an object.
        offset_x, offset_y = self.get_text_position_offsets(object_size)
        center_x = (point1[0] + point2[0]) // 2
        center_y = (point1[1] + point2[1]) // 2
        cv2.putText(img=self.cvimg['sized'],
                    text=object_size,
                    org=(round(center_x - offset_x),
                         round(center_y + offset_y)),
                    fontFace=const.FONT_TYPE,
                    fontScale=self.font_scale,
                    color=text_contrast,
                    thickness=self.line_thickness,
                    lineType=cv2.LINE_AA,
                    )

    def display_all_objects(self, event=None) -> Event:
        """
        Draw all annotated objects in the image, with their size, in a
        bounding box rectangle.
        Called from process_prediction().
        Calls convert_bbox_data(), annotate_object(), update_image().

        Args:
            event: Used for any implicit tkinter event.
        Returns:
            Event, a formality to pass IDE inspections.
        """

        self.cvimg['sized'] = self.cvimg['input'].copy()

        # Clear the list of sized oysters used in reporting metrics for
        #  when a new image or a new confidence limit is processed.
        #  Clearing is not needed for calls from view-related methods,
        #  but doing so keeps things simple w/o affecting performance.
        self.oyster_sizes.clear()
        def _display_objects(objects: np.ndarray, name: str) -> None:
            for bbox in objects:
                x1, y1, x2, y2, display_size = self.convert_bbox_data(bbox)
                if name == 'oyster':
                    oyster_size = display_size.rstrip('px')
                    self.oyster_sizes.append(float(oyster_size))

                self.annotate_object(point1=(x1, y1),
                                     point2=(x2, y2),
                                     object_size=display_size,
                                     object_name=name)

        _display_objects(self.true_pos_oysters, name='oyster')
        _display_objects(self.true_pos_standards, name='standard')
        self.update_image('sized')
        return event

    def display_metrics_in_image(self) -> None:
        """
        Display the image metrics in a text box at the top left of the
        Sized Objects image.
        Called from start_now(), setup_main_menu(), bind_functions(),
        process_prediction(), config_entries().
        Calls get_text_position_offsets(), to_precision(), update_image().
        Returns: None
        """

        # Call display_all_objects() to ensure that the inserted text box
        #  is redrawn from the original image, not from the last annotated.
        #  This prevents the alpha overlay from being applied multiple times.
        self.display_all_objects()

        _sf: int = self.get_sig_fig()
        _cf: float = utils.get_correction_factor(self.bbox_ratio_mean)

        if self.oyster_sizes:
            mean_size: float = mean(self.oyster_sizes) * _cf if _cf > 1.0 else self.oyster_sizes[0]
            if self.entry['size_std_val'].get() == '1':
                mean_oyster_size = str(int(mean_size))
            else:
                mean_oyster_size = (f'{to_p.to_precision(value=mean_size, precision=_sf)};'
                                    f' {_sf} sig figs')
        else:
            mean_oyster_size = 'n/a'

        display_metrics = (
            f'Image: {self.input_file_name}\n'
            f'Avg Size: {mean_oyster_size}\n'
            f'Counted: {len(self.oyster_sizes)}\n'
        )

        longest_line: str = max(display_metrics.split('\n'), key=len)
        x_offset, y_offset = self.get_text_position_offsets(longest_line)
        textbox_px_width = round(x_offset * 2.2)
        img_height = self.cvimg['input'].shape[1]

        # Template for transparent white text box:
        #  https://pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        overlay = self.cvimg['sized'].copy()
        cv2.rectangle(img=overlay,
                      pt1=(5, 5),
                      pt2=(textbox_px_width, img_height // 12),
                      color=const.COLORS_CV['white'],
                      thickness=cv2.FILLED,
        )
        cv2.addWeighted(src1=overlay,
                        alpha=const.ALPHA,
                        src2=self.cvimg['sized'].copy(),  # another copy, to avoid overwriting.
                        beta=1 - const.ALPHA,
                        gamma=0.0,
                        dst=self.cvimg['sized']
        )

        # Need to put one line at a time to avoid overlapping text.
        # org: bottom-left corner of the text annotation.
        for i, line in enumerate(display_metrics.split('\n'), start=1):
            _y = i * img_height // 50 + y_offset
            cv2.putText(img=self.cvimg['sized'],
                        org=(10, round(_y)),  # add 5 to the x indent of cv2.rectangle pt1.
                        text=line,
                        fontFace=const.FONT_TYPE,
                        fontScale=self.font_scale,
                        color=const.COLORS_CV['black'],
                        thickness=self.line_thickness,
                        lineType=cv2.LINE_AA,
            )

        self.update_image('sized')

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from start_now(), process_prediction(), process_sizes().
        Calls get_sig_fig(), to_precision(), utils.display_report().

        Returns: None
        """

        size_std_dia = ('1, sizes are in pixels'
                        if self.entry['size_std_val'].get() == '1'
                        else self.entry['size_std_val'].get()
                        )
        num_std_objects = len(self.true_pos_standards)
        num_oysters = len(self.true_pos_oysters)

        sig_fig: int = self.get_sig_fig()
        avg_std_size: str = to_p.to_precision(
            value=str(self.standards_mean_measured_size.get()),
            precision=sig_fig)

        # Need this hack when a new image is opened, but instead of clicking
        #  "Process or Update", the user again clicks "New input"
        #  followed by 'Cancel' before clicking "Process or Update".
        #  This is a rare case, but results in input_file_path being empty.
        #  The processed input_file_name is retained, so report that.
        input_file = self.input_file_name if not self.input_file_path else self.input_file_path

        # Work up some summary metrics with correct number of sig. fig.
        #  and estimated corrected oyster size metrics.
        # When displaying sizes as pixels, don't apply sig. fig.
        # if self.oyster_sizes and num_oysters > 0 and self.interior_standards.size:
        if self.oyster_sizes and self.interior_standards.size:
            _cf: float = utils.get_correction_factor(self.bbox_ratio_mean)
            mean_size: float = mean(self.oyster_sizes) * _cf if _cf > 1.0 else self.oyster_sizes[0]
            median_size: float = median(self.oyster_sizes) * _cf if num_oysters > 1 else self.oyster_sizes[0]
            if self.entry['size_std_val'].get() == '1':
                mean_oyster_size = str(int(mean_size))
                median_oyster_size = str(int(median_size))
                smallest = str(int(min(self.oyster_sizes)))
                biggest = str(int(max(self.oyster_sizes)))
            else:
                mean_oyster_size: str = to_p.to_precision(value=mean_size, precision=sig_fig)
                median_oyster_size: str = to_p.to_precision(value=median_size, precision=sig_fig)
                smallest: str = to_p.to_precision(value=min(self.oyster_sizes), precision=sig_fig)
                biggest: str = to_p.to_precision(value=max(self.oyster_sizes), precision=sig_fig)

            median_oyster_txt = (f'{median_oyster_size} (corrected: {"+" if _cf > 1.0 else ""}'
                                     f'{round((_cf - 1) * 100, 1)}%)')
            size_range: str = f'{smallest}--{biggest}'
        elif not self.interior_standards.size:
            mean_oyster_size = median_oyster_txt = size_range = 'n/a'
            avg_std_size = 'n/a'
        else:  # standards found, but no oysters found.
            mean_oyster_size = median_oyster_txt = size_range = 'n/a'

        # Text is formatted for clarity in window, terminal, and saved file.
        # Divider symbol is Box Drawings Double Horizontal from https://coolsymbol.com/
        # Divider's unicode_escape: u'\u2550\' or u'\u2550'
        # Report oyster mean and median size corrected for box ratio mean,
        #  but report range as uncorrected to match the image display.
        space = 26
        tab = " " * space
        divider = "═" * 20

        self.report_txt = (
            f'\nImage: {input_file}\n'
            f'Image size, pixels (w x h): {self.input_w}x{self.input_ht}\n'
            f'{divider}\n'
            f'{"Confidence level (%):".ljust(space)}{self.confidence_slide_val.get()}\n'
            f'{"# oysters:".ljust(space)}{num_oysters}\n'
            f'{"# standards:".ljust(space)}{num_std_objects}\n'
            f'{"Entered standard size:".ljust(space)}diameter = {size_std_dia}\n'
            f'{"Avg. standard size used:".ljust(space)}{avg_std_size}\n'
            f'{"Oyster sizes:".ljust(space)}average = {mean_oyster_size},'
            f' median = {median_oyster_txt},\n'
            f'{tab}range = {size_range} (uncorrected)'
        )

        utils.display_report(frame=self.report_frame,
                             report=self.report_txt)

    def display_processing_info(self) -> None:
        """
        Display an informational or warning message in the main window
        for sizing and processing operations.
        Called from process_prediction().
        Calls show_info_message(), get_standard_sizes().
        """

        # Elements are in order of condition priority.
        #  The first true condition will break the loop and be the one
        #  displayed in the info_label. The color for message text in
        #  show_info_message() is the second element of the message tuple.
        processing_info_messages = (
            (self.true_pos_standards.size and (self.get_standard_sizes()).std() > 10, (
                'Detected standards (purple box) are different sizes.\n'
                'Sizing results may be inaccurate.\n'
                'Consider adjusting the Confidence level.\n',
                "vermilion")),
            (len(self.predicted_boxes) >= const.PREDICT_MAX_DET, (
                f'DETECTION LIMIT of {const.PREDICT_MAX_DET} WAS MET.\n'
                'Valid objects may have been excluded.\n'
                'Sizing results may be inaccurate.\n'
                'Consider increasing the Confidence level.',
                "vermilion")),
            (len(self.true_pos_oysters) < len(self.interior_oysters), (
                'Overlapping false positives were found and removed.\n'
                'Increase Confidence level if no size standard detected.\n\n',
                "vermilion")),
            (len(self.true_pos_standards) < len(self.interior_standards), (
                'Overlapping false positives were found and removed.\n'
                'Increasing Confidence level may improve results.\n\n',
                "vermilion")),
            (self.first_run, (
                f'Initial processing time elapsed: {self.elapsed}\n'
                'Identified size standard have a purple box.\n'
                'Adjust Confidence level if any oysters have a purple box.\n',
                "black")),
            (not self.first_run, (
                'Object detections completed.\n'
                f'{self.elapsed} processing seconds elapsed.\n'
                'Identified size standard have a purple box.\n'
                'Adjust Confidence level if any oysters have a purple box.\n',
                "blue")),
        )
        for condition, message in processing_info_messages:
            if condition:
                self.show_info_message(info=message[0], color=message[1])
                break

    def process_sizes(self) -> None:
        """
        Process the sizes of oysters the image, for when a size standard
        value is entered. Display and report the results.
        Called from process_prediction() and from config_entries() as
        an Entry() binding.

        Returns: None
        """

        self.find_interior_objects()
        self.find_true_pos_objects()
        self.set_bbox_ratio_mean(bbox_ary=self.true_pos_oysters)
        self.validate_size_entry()
        self.determine_mean_standard_size()
        self.get_sig_fig()
        self.display_all_objects()
        self.report_results()

    def process_prediction(self, event=None) -> Event:
        """
        Calls methods Process_image.prediction() and process_sizes(),
        which in turn calls methods for filtering, measuring, annotating,
        and reporting.
        Called from start_now() and various callbacks and bindings.

        Args:
            event: Used for any implicit tkinter event.
        Returns:
            Event, a formality to pass IDE inspections.
        """

        # If no objects found, then no need to update beyond prediction().
        # Record processing time to display in info_txt. When no objects
        #  are found, the elapsed time is considered n/a.
        # The oyster_sizes list is cleared when no objects are found,
        #  otherwise it would retain the last run's sizes, which is
        #  normally cleared in display_all_objects() via process_sizes().
        self.widget_control(action='off')
        self.time_start: float = time()
        self.prediction()
        if self.predicted_boxes.size:
            self.process_sizes()
            self.elapsed =(round(time() - self.time_start, 3)
                           if (self.interior_standards.size and
                               self.interior_oysters.size)
                           else 'n/a')
        else:
            utils.no_objects_found_msg(caller='predicted_boxes')
            self.oyster_sizes.clear()
            self.report_results()
            self.cvimg['sized'] = self.cvimg['input'].copy()
            self.update_image('sized')
            self.elapsed = 'n/a'

        self.widget_control(action='on')
        self.display_metrics_in_image()
        self.display_processing_info()
        return event  # a formality to pass IDE inspections


class SetupApp(ViewImage):
    """
    The mainloop Class for file handling and configuring windows and widgets.
    Methods:
    call_cmd
    setup_main_window
    setup_main_menu
    start_now
    open_input
    close_window_message
    setup_image_windows
    configure_main_window
    configure_buttons
    config_entries
    config_sliders
    bind_focus_actions
    bind_functions
    set_defaults
    grid_widgets
    grid_img_labels
    display_images
    """

    def __init__(self):
        super().__init__()

        # Dictionary items are populated in setup_image_windows(), with
        #   tk.Toplevel as values; don't want tk windows created here.
        self.tkimg_window: dict = {}

        # Attributes defined in setup_main_menu().
        self.menubar = tk.Menu()
        self.menu_labels: tuple = ()

    def call_cmd(self) -> '_Command':
        """
        Groups methods that are shared by buttons, menus, and
        key bind commands in a nested Class.
        Called from setup_main_window(), add_menu_bar(),
        bind_functions(), bind_scale_adjustment(),
        configure_buttons().
        Usage example: self.call_cmd().save_results()

        Returns: A callable method from the _Command inner class.
        """

        # Inner class concept adapted from:
        # https://stackoverflow.com/questions/719705/
        #   what-is-the-purpose-of-pythons-inner-classes/722175
        cv_colors = list(const.COLORS_CV.keys())

        def _display_annotation_action(action: str, value: str):
            self.show_info_message(
                'A new annotation style was applied.\n'
                f'{action} was changed to {value}.\n',
                color='black')

        def _display_scale_action(value: float):
            """
            The scale_factor is applied in ProcessImage.update_image()
            Called from _Command.increase_scale, _Command.decrease_scale.

            Args:
                 value: the scale factor update to display, as float.
            """
            _sf = round(value, 2)
            self.show_info_message(
                f'A new scale factor of {_sf} was applied.\n\n',
                color='black')

            for _title in const.WINDOW_TITLES:
                self.update_image(img_name=_title)


        class _Command:
            """
            Gives command-based methods access to all script methods and
            instance variables.
            Methods:
            save_results
            new_input
            increase_font_size
            decrease_font_size
            increase_line_thickness
            decrease_line_thickness
            next_font_color
            preceding_font_color
            increase_scale_factor
            decrease_scale_factor
            apply_default_settings
            """

            # These methods are called from configure_buttons(), key
            #  bindings and the "File" menubar of add_menu_bar().
            @staticmethod
            def save_results():
                """
                Save annotated sized image and its Report text with
                individual object sizes appended.
                Calls utils.save_report_and_img(), show_info_message().
                Called from keybinding, menu, and button commands.
                """
                _sizes = ', '.join(str(i) for i in self.oyster_sizes)
                utils.save_report_and_img(
                    path2folder=self.input_file_path,
                    img2save=self.cvimg['sized'],
                    txt2save=self.report_txt + f'\n{_sizes}',
                    caller=PROGRAM_NAME,
                )
                self.show_info_message(
                    'Results report and annotated image were saved to\n'
                    f'the input image folder: {self.input_folder_name}\n',
                    color='blue')

            @staticmethod
            def new_input():
                """
                Reads a new image file for preprocessing.
                Calls open_input(), set_auto_scale_factor(), update_image().
                Called from keybinding, menu, and button commands.

                Returns: None
                """

                if self.open_input(parent=self.master):
                    self.set_auto_scale_factor()
                    self.update_image(img_name='input')
                else:  # User canceled input selection or closed messagebox window.
                    self.show_info_message(
                        'No new input file was selected.\n\n',
                        color='vermilion')

            # These methods are called from the "Style" menu of add_menu_bar()
            #  and as bindings from setup_main_window() or bind_functions().
            @staticmethod
            def increase_font_size() -> None:
                """Limit upper font size scale to a 3x increase."""
                self.font_scale *= 1.1
                self.font_scale = round(min(self.font_scale, 3), 2)
                self.display_all_objects()
                _display_annotation_action('Font scale', f'{self.font_scale}')

            @staticmethod
            def decrease_font_size() -> None:
                """Limit lower font size scale to a 1/3 decrease."""
                self.font_scale *= 0.9
                self.font_scale = round(max(self.font_scale, 0.33), 2)
                self.display_all_objects()
                _display_annotation_action('Font scale', f'{self.font_scale}')

            @staticmethod
            def increase_line_thickness() -> None:
                """Limit upper thickness to 15."""
                self.line_thickness += 1
                self.line_thickness = min(self.line_thickness, 15)
                self.display_all_objects()
                _display_annotation_action('Line thickness', f'{self.line_thickness}')

            @staticmethod
            def decrease_line_thickness() -> None:
                """Limit lower thickness to 1."""
                self.line_thickness -= 1
                self.line_thickness = max(self.line_thickness, 1)
                self.display_all_objects()
                _display_annotation_action('Line thickness', f'{self.line_thickness}')

            @staticmethod
            def next_font_color() -> None:
                """Go to the next color key in const.COLORS_CV.keys."""
                current_color: str = self.color_val.get()
                current_index = cv_colors.index(current_color)
                # Wraps around the list to the first color.
                next_color = (cv_colors[0]
                              if current_index == len(cv_colors) - 1
                              else cv_colors[current_index + 1])
                self.color_val.set(next_color)
                self.display_all_objects()
                _display_annotation_action('Font color', f'{next_color}')

            @staticmethod
            def preceding_font_color() -> None:
                """Go to the prior color key in const.COLORS_CV.keys."""
                current_color: str = self.color_val.get()
                current_index = cv_colors.index(current_color)
                # Wraps around the list to the last color.
                preceding_color = (cv_colors[len(cv_colors) - 1]
                                   if current_index == 0
                                   else cv_colors[current_index - 1])
                self.color_val.set(preceding_color)
                self.display_all_objects()
                _display_annotation_action('Font color', f'{preceding_color}')

            @staticmethod
            def increase_scale_factor() -> None:
                """
                Limit upper factor to a 5x increase to maintain performance.
                """
                scale_factor: float = self.scale_factor.get()
                self.scale_factor.set(round(min(scale_factor * 1.1, 5), 2))
                _display_scale_action(value=scale_factor)

            @staticmethod
            def decrease_scale_factor() -> None:
                """
                Limit lower factor to a 1/10 decrease to maintain readability.
                """
                scale_factor: float = self.scale_factor.get()
                self.scale_factor.set(round(max(scale_factor * 0.9, 0.10), 2))
                _display_scale_action(value=scale_factor)

            # This method not currently used.
            @staticmethod
            def apply_default_settings():
                """
                Resets settings values and processes images.
                Calls set_auto_scale_factor(), set_defaults(), process_prediction(), and
                show_info_message().
                Called from keybinding, menu, and button commands.
                """

                # Order of calls is important here.
                self.set_auto_scale_factor()
                self.set_defaults()
                self.widget_control('off')  # is turned 'on' in process_prediction()
                self.show_info_message(
                    'Settings have been reset to their defaults.\n'
                    'Check and adjust if needed.\n', color='blue')
                self.process_prediction()

        return _Command

    def setup_main_window(self):
        """
        For clarity, remove from view the Tk mainloop window created
        by the inherited ProcessImage() Class. But, to make window
        transitions smoother, first position it where it needs to go.
        Put this window toward the top right corner of the screen
        so that it doesn't cover up the img windows; also so that
        the bottom of the window is, hopefully, not below the bottom
        of the screen.

        Returns:
            None
        """

        self.screen_width = self.winfo_screenwidth()

        # Make geometry offset a function of the screen width.
        #  This is needed b/c of the way different platforms' window
        #  managers position windows.
        w_offset = int(self.screen_width * 0.55)
        self.geometry(f'+{w_offset}+50')
        win_ht = 340 if MY_OS == 'dar' else 370
        self.wm_minsize(width=500, height=win_ht)

        # Need to allow complete tk mainloop shutdown from the system's
        #  window manager 'close' icon in the start window bar. And
        #  provide Terminal exit message.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(mainloop=self))


    def setup_main_menu(self) -> None:
        """
        Create main (app) menu instance and hierarchical menus.
        For proper menu functions, must be called in main(), and after
        setup_main_window(). Calls style functions in call_cmd().

        Returns: None
        """

        # Accelerators use key binds from bind_functions() and
        #   bind_functions() and must be platform-specific.
        # Unicode arrow symbols: left \u2190, right \u2192, up \u2101, down \u2193
        os_accelerator = 'command' if MY_OS == 'dar' else 'Ctrl'
        style_accelerator = 'control' if MY_OS == 'dar' else 'Ctrl'
        zoom_accelerator = 'command-control' if MY_OS == 'dar' else 'Ctrl'
        color_tip = ('command-↑ & command-↓'
                     if MY_OS == 'dar'
                     else 'Ctrl-↑ & Ctrl-↓')
        zoom_tip = ('with command-control-← & command-control-→.'
                    if MY_OS == 'dar'
                    else 'with Ctrl-← & Ctrl-→.')
        plus_key, minus_key = ('+', '-') if MY_OS == 'dar' else ('(plus)', '(minus)')

        menu_params = dict(
            tearoff=0,
            takefocus=False,
            type='menubar',
            font=const.MENU_FONT,
        )

        # Note: menu_labels is also used in bind_focus_actions().
        self.menu_labels = ('File', 'Style', 'View', 'Help')
        menu = {_l: tk.Menu(**menu_params) for _l in self.menu_labels}

        for _l in self.menu_labels:
            self.menubar.add_cascade(label=_l, menu=menu[_l])

        menu['File'].add_command(label='Save results',
                                 command=self.call_cmd().save_results,
                                 accelerator=f'{os_accelerator}+S')
        menu['File'].add_command(label='New input...',
                                 command=self.call_cmd().new_input,
                                 accelerator=f'{os_accelerator}+N')
        menu['File'].add(tk.SEPARATOR)
        menu['File'].add_command(label='Quit',
                                 command=lambda: utils.quit_gui(mainloop=self),
                                 # macOS doesn't recognize 'Command+Q' as an accelerator
                                 #   b/c cannot override that system's native Command-Q,
                                 accelerator=f'{os_accelerator}+Q')

        menu['Style'].add_command(label='Increase font size',
                                  command=self.call_cmd().increase_font_size,
                                  accelerator=f'{style_accelerator}+{plus_key}')
        menu['Style'].add_command(label='Decrease font size',
                                  command=self.call_cmd().decrease_font_size,
                                  accelerator=f'{style_accelerator}+{minus_key}')
        menu['Style'].add_command(label='Increase line thickness',
                                  command=self.call_cmd().increase_line_thickness,
                                  accelerator=f'Shift+{style_accelerator}+{plus_key}')
        menu['Style'].add_command(label='Decrease line thickness',
                                  command=self.call_cmd().decrease_line_thickness,
                                  accelerator=f'Shift+{style_accelerator}+{minus_key}')
        menu['Style'].add_command(label='Next color',
                                  command=self.call_cmd().next_font_color,
                                  accelerator=f'{zoom_accelerator}+↑')
        menu['Style'].add_command(label='Prior color',
                                  command=self.call_cmd().preceding_font_color,
                                  accelerator=f'{zoom_accelerator}+↓')
        menu['View'].add_command(label='Zoom images out',
                                 command=self.call_cmd().decrease_scale_factor,
                                 accelerator=f'{zoom_accelerator}+←')
        menu['View'].add_command(label='Zoom images in',
                                 command=self.call_cmd().increase_scale_factor,
                                 accelerator=f'{zoom_accelerator}+→')
        menu['View'].add_command(label='Update Sized Objects window',
                                 command=self.process_prediction,
                                 accelerator=f'{zoom_accelerator}+U')
        menu['View'].add_command(label='Display inset in Sized Objects image',
                                 command=self.display_metrics_in_image,
                                 accelerator=f'{zoom_accelerator}+I')

        tips = tk.Menu(**menu_params)
        menu['Help'].add_cascade(label='Tips...', menu=tips)
        # Bullet symbol from https://coolsymbol.com/, unicode_escape: u'\u2022'
        tip_text = (
            '• Images are auto-zoomed to fit the screen at startup.',
            f'     Zoom can be changed with {zoom_tip}',
            f'• Box color can be changed with {color_tip}, except',
            "      standard's box is always purple.",
            '• Font size can be changed with Ctrl-+(plus) & -(minus).',
            '• Boldness can be changed with Shift-Ctrl-+(plus) & -(minus).',
            '• Ctrl-U updates the sized image window for a new image.',
            '• Esc or Ctrl-Q from any window will exit the program.',
            "• More Tips are in the repository's README file.",
        )
        for _line in tip_text:
            tips.add_command(label=_line, font=const.TIPS_FONT)

        menu['Help'].add_command(label='About',
                                 command=utils.about_window)

    def start_now(self) -> None:
        """
        Initiate the processing pipeline by setting up and configuring
        all widgets.
        Called from main() at script start.
        Returns:
            None
        """

        # This calling sequence ensures that everything displays
        #  as expected with a visually clean start.
        # For proper menu functions, setup_main_menu() must be called
        #  after setup_main_window() and setup_main_window() must be
        #  called first.
        # process_prediction() is inherited from ViewImage(), others
        #  are methods of SetupApp().
        utils.set_icon(self)
        self.setup_main_window()
        self.setup_main_menu()
        self.open_input(parent=self.master if MY_OS == 'dar' else self)
        self.set_auto_scale_factor()
        self.setup_image_windows()
        self.configure_main_window()
        self.report_results()
        self.display_metrics_in_image()
        self.configure_buttons()
        self.config_entries()
        self.config_sliders()
        self.bind_functions()
        self.bind_focus_actions()
        self.set_defaults()
        self.grid_widgets()
        self.grid_img_labels()
        self.process_prediction()
        self.display_images()
        self.first_run = False

    def open_input(self, parent: Union[tk.Toplevel, 'SetupApp']) -> bool:
        """
        Provides an open file dialog to select a starting or new input
        image file. Also sets a scale slider value for the displayed img.
        Called from start_now() and call_cmd().new_input.
        Args:
            parent: The window or mainloop Class over which to place the
                file dialog, e.g., app, self, or self.master (macOS).

        Returns:
            True or False depending on whether input was selected.

        """
        self.input_file_path = filedialog.askopenfilename(
            parent=parent,
            title='Select input image',
            filetypes=[('JPG', '*.jpg'),
                       ('JPG', '*.jpeg'),
                       ('JPG', '*.JPEG'),
                       ('JPG', '*.JPG'),  # used for iPhone images
                       ('PNG', '*.png'),
                       ('PNG', '*.PNG'),
                       ('All', '*.*')],
        )

        # When user selects an input, check whether it can be used by OpenCV.
        #  If so, open it, and proceed. If user selects "Cancel" instead of
        #  selecting a file, then quit if at start up, otherwise close the
        #  filedialog (default action) because this was called from the
        #  "New input" button in the mainloop tk toplevel window (self.master).
        # Need to call quit_gui() without confirmation b/c a confirmation
        #  dialog answer of "No" throws an error during file input.
        try:
            if self.input_file_path:
                self.cvimg['input'] = cv2.imread(self.input_file_path)
                self.input_ht, self.input_w, _ = self.cvimg['input'].shape
                self.input_file_name = Path(self.input_file_path).name
                self.input_folder_name = Path(self.input_file_path).parent.name
                self.show_info_message(info=f'{self.input_file_name} loaded.\n'
                                            f'Press Ctrl-U or "Process" to update\n',
                                       color='blue')
            elif parent != self.master:  # at startup file dialog, so quit.
                print('Invalid or no file selected. Quitting...')
                utils.quit_gui(mainloop=self, confirm=False)
            else:  # at main window, app tk.Toplevel, so try again.
                return False
        # Can get attribute error from other functions when cvimg['input'] is None.
        except (cv2.error, AttributeError):
            msg = f'{self.input_file_path} cannot be used.'
            if not self.first_run:
                messagebox.showerror(
                    title="Bad input file",
                    detail=msg + '\nUse "New input" to try another file.')
                return False
            else:
                print(f'{msg}')
                messagebox.showerror(
                    title="Bad input file",
                    detail=msg + '\nRestart and try a different file.\nQuitting...')
                utils.quit_gui(mainloop=self, confirm=False)

        # Auto-set images' scale factor based on input image px dimensions.
        #  Can be later reset with keybindings in bind_scale_adjustment().
        self.metrics = manage.input_metrics(img=self.cvimg['input'])
        self.line_thickness = self.metrics['line_thickness']  # img size dependent
        # self.line_thickness = utils.set_line_thickness(self)  # screen size dependent
        self.font_scale = self.metrics['font_scale']
        return True

    def close_window_message(self) -> None:
        """
        Provide a notice in Report and Settings (mainloop, self)
        window.
        Called only as a .protocol() func in setup_image_windows().

        Returns: None
        """

        prev_txt = self.info_txt.get()
        prev_fg = self.info_label.cget('fg')

        self.show_info_message(
            info='That window cannot be closed from its window bar.\n'
                 'Minimize it if it is in the way.\n'
                 'Esc or Ctrl-Q keys can quit the program.\n',
            color='vermilion')

        self.update_idletasks()

        # Give user time to read the _info before resetting it to
        #  the previous info text.
        self.after(7777,self.show_info_message,prev_txt, prev_fg)

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels that
        are used to display and update processed images.
        Called from start_now().
        """

        # Dictionary window_title item order determines stack order of windows.
        # Toplevel() is assigned here, not in __init__, to control timing
        #  and smoothness of window appearance at startup.
        # Labels to display scaled images are updated using .configure()
        #  for 'image=' in ViewImage.update_image().
        #  Labels are gridded in their respective tkimg_window in grid_img_labels().
        # Need custom icon to replace default tk desktop icon for each img window.
        # Withdraw sized window here for clean transition; it is deiconified
        #  in display_images().
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #  images will remain anchored at their top left corners.
        self.tkimg_window = {_t: tk.Toplevel() for _t in const.WINDOW_TITLES}
        self.img_label = {_t: tk.Label(self.tkimg_window[_t]) for _t in const.WINDOW_TITLES}
        for _title, _toplevel in self.tkimg_window.items():
            utils.set_icon(_toplevel)
            _toplevel.wm_minsize(width=200, height=100)
            _toplevel.resizable(width=False, height=False)
            _toplevel.protocol(name='WM_DELETE_WINDOW', func=self.close_window_message)
            _toplevel.columnconfigure(index=0, weight=1)
            _toplevel.columnconfigure(index=1, weight=1)
            _toplevel.rowconfigure(index=0, weight=1)
            _toplevel.title(const.WINDOW_TITLES[_title])
            _toplevel.config(**const.WINDOW_PARAMETERS)
            self.update_image(img_name=_title)
            if _title == 'sized':
                _toplevel.withdraw()

    def configure_main_window(self) -> None:
        """
        Settings and report window (mainloop, self) keybindings,
        configurations, and grids for settings and reporting frames.
        Called from start_now().
        """

        self.config(**const.WINDOW_PARAMETERS)
        self.config(bg=const.MASTER_BG)
        self.config(menu=self.menubar)

        # Default Frame() arguments work fine to display report text.
        # bg won't show when grid sticky EW for tk.Text; see utils.display_report().
        self.selectors_frame.configure(relief='raised',
                                       bg=const.DARK_BG,
                                       # bg=const.COLORS_TK['sky blue'],  # for development
                                       borderwidth=5)

        self.columnconfigure(index=0, weight=1)
        self.columnconfigure(index=1, weight=1)
        self.report_frame.columnconfigure(index=0, weight=1)

        # Allow only sliders, not their labels, to expand with window.
        self.selectors_frame.columnconfigure(index=1, weight=1)

        self.report_frame.grid(column=0, row=0,
                               columnspan=2,
                               padx=(5, 5), pady=(5, 5),
                               sticky=tk.EW)
        self.selectors_frame.grid(column=0, row=1,
                                  columnspan=2,
                                  padx=5, pady=(0, 5),
                                  ipadx=4, ipady=4,
                                  sticky=tk.EW)

        # Width should fit any text expected without causing WINDOW shifting.
        self.info_label.config(font=const.TIPS_FONT,
                               width=50,  # width should fit any text expected without
                               justify='right',
                               bg=const.MASTER_BG,  # use 'pink' for development
                               fg='black')

    def configure_buttons(self) -> None:
        """
        Assign and grid Buttons in the settings (mainloop, self) window.
        Called from start_now().
        """
        manage.ttk_styles(mainloop=self)

        # Configure all items in the dictionary of ttk buttons.
        button_params = dict(
            width=0,
            style='My.TButton',
        )

        self.button['update'].config(
            text='Process or Update',
            command=self.process_prediction,
            **button_params,
        )

        self.button['save_results'].config(
            text='Save results',
            command=self.call_cmd().save_results,
            **button_params,
        )

        self.button['new_input'].config(
            text='New input',
            command=self.call_cmd().new_input,
            **button_params,
        )

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop, self) window.
        Called from start_now().
        """

        def _check_and_process(event: tk.Event) -> Event:
            self.process_sizes()
            self.display_metrics_in_image()
            return event

        self.entry['size_entry'].config(textvariable=self.entry['size_std_val'], width=8)
        self.entry['size_std_lbl'].config(text="Enter standard's diameter:",
                                          **const.LABEL_PARAMETERS)
        self.entry['size_std_lbl2'].config(text="Entry of 1 provides pixel sizes.",
                                           **const.LABEL_PARAMETERS)

        self.entry['size_entry'].bind('<Return>', _check_and_process)
        self.entry['size_entry'].bind('<KP_Enter>', _check_and_process)

    def config_sliders(self) -> None:
        """
        Configure arguments and mouse button bindings for all Scale
        widgets in the settings (mainloop, self) window.
        Called from start_now().
        """
        # Minimum width for any Toplevel window is set by the length
        #  of the longest widget, whether that be a Label() or Scale().
        #  So, for the main (app) window, set a Scale() length sufficient
        #  to fit everything in the Frame given current padding arguments.
        #  Keep in mind that a long input file path in the report_frame
        #   may be longer than this set scale_len in the selectors_frame.
        scale_len = int(self.screen_width * 0.20)

        self.slider['confidence_lbl'].configure(text='Confidence level, %:\n',
                                                **const.LABEL_PARAMETERS,
                                                )
        self.slider['confidence'].configure(from_=50, to=100,
                                            tickinterval=5,
                                            length=scale_len,
                                            variable=self.confidence_slide_val,
                                            **const.SCALE_PARAMETERS,
                                            )
        # To avoid processing all the intermediate values between normal
        #  slider movements, bind slider to call function only on
        #  left button release.
        self.slider['confidence'].bind('<ButtonRelease-1>', func=self.process_prediction)

    def bind_focus_actions(self) -> None:
        """
        Configure menu bar headings with normal color when main has focus
        and grey-out when not. Called at startup from start_now().
        """

        def _got_focus(_) -> None:
            """The '_' is a placeholder for an event parameter."""
            try:
                for label in self.menu_labels:
                    self.menubar.entryconfig(index=label, state=tk.NORMAL)
            except tk.TclError:
                print('ignoring macOS Tcl entryconfig focus error')

        def _lost_focus(_) -> None:
            for label in self.menu_labels:
                self.menubar.entryconfig(index=label, state=tk.DISABLED)

        # Because we are retaining the macOS default menu bar, the menu
        #  headings are not greyed out when the main window loses focus,
        #  and remain active when any window has focus.
        if MY_OS == 'dar':
            self.bind_all('<FocusIn>', _got_focus)
            self.bind_all('<FocusOut>', _lost_focus)
        else:
            self.bind('<FocusIn>', _got_focus)
            self.bind('<FocusOut>', _lost_focus)

    def bind_functions(self) -> None:
        """
        Set key bindings to change annotation styles, and to process,
        save, and import images.
        Called from start_now().
        Calls methods from the inner _Command class of self.call_cmd().
        """

        self.bind_all('<Escape>', lambda _: utils.quit_gui(mainloop=self))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(mainloop=self))

        # NOTE: In Windows, KP_* is not a recognized keysym string; works on Linux.
        #  Windows keysyms 'plus' & 'minus' are for both keyboard and keypad.
        # Note: macOS Command-q will quit program without utils.quit_gui info msg.
        # Need os-specific control key bindings for macOS and Windows/Linux.
        event_function = {
            f'<{f"{const.C_BIND}"}-u>': self.process_prediction,
            f'<{f"{const.C_BIND}"}-s>': self.call_cmd().save_results,
            f'<{f"{const.C_BIND}"}-n>': self.call_cmd().new_input,
            f'<{f"{const.C_BIND}"}-i>': self.display_metrics_in_image,
            '<Control-equal>': self.call_cmd().increase_font_size,
            '<Control-minus>': self.call_cmd().decrease_font_size,
            '<Control-KP_Subtract>': self.call_cmd().decrease_font_size,
            '<Shift-Control-plus>': self.call_cmd().increase_line_thickness,
            '<Shift-Control-KP_Add>': self.call_cmd().increase_line_thickness,
            '<Shift-Control-underscore>': self.call_cmd().decrease_line_thickness,
            '<Control-Up>': self.call_cmd().next_font_color,
            '<Control-Down>': self.call_cmd().preceding_font_color,
            '<Control-Right>': self.call_cmd().increase_scale_factor,
            '<Control-Left>': self.call_cmd().decrease_scale_factor,
        }

        # Some bindings are needed only for the settings window, but it is
        #  simpler to use bind_all(), which does not depend on widget focus.
        for event, function in event_function.items():
            self.bind_all(event, lambda _, f=function: f())

        # Need additional platform-specific keypad keysyms.
        if MY_OS == 'win':
            self.bind_all('<Control-plus>',
                          lambda _: self.call_cmd().increase_font_size())
            self.bind_all('<Shift-Control-minus>',
                          lambda _: self.call_cmd().decrease_line_thickness())
        else:  # is Linux or macOS
            self.bind_all('<Control-KP_Add>',
                          lambda _: self.call_cmd().increase_font_size())
            self.bind_all('<Shift-Control-KP_Subtract>',
                          lambda _: self.call_cmd().decrease_line_thickness())

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets and keybinds for processing and
        annotating. Called from start_now().
        Returns: None
        """

        self.color_val.set('gold1')
        self.entry['size_std_val'].set('1')
        self.confidence_slide_val.set(80)  # 0.80 confidence was used for training.

    def grid_widgets(self) -> None:
        """
        Developer: Grid all widgets here, as a method, to clarify spatial
        relationships.
        Called from start_now().
        """

        # Use the dict() function with keyword arguments to visually mimic
        #  the keyword parameter structure of grid().
        west_grid_params = dict(
            padx=5,
            pady=(0, 5),
            sticky=tk.W,
        )

        button_grid_params = dict(
            padx=10,
            pady=(0, 2),
            sticky=tk.W,
        )

        # info_label widget is in the main window (self).
        # Note: rowspan=3 works well with the  3-4 return characters
        #  in each info string to prevent shifts of frame row spacing.
        #  Use 3 because that seems to be needed to cover the combined
        #  height of the last three main window rows (2, 3, 4) with buttons.
        #  Sticky is 'east' to prevent horizontal shifting when, during
        #  processing, all buttons in col 0 are removed.
        self.info_label.grid(column=1, row=2, rowspan=3, columnspan=2,
                             padx=(0, 5), sticky=tk.E,
                             )

        # Widgets gridded in the self.selectors_frame Frame.
        # Sorted by row number
        self.slider['confidence_lbl'].grid(column=0, row=0,
                                           padx=(5, 10), pady=(10, 5), sticky=tk.W,
                                           )

        self.entry['size_std_lbl'].grid(column=0, row=1, **west_grid_params)
        self.entry['size_std_lbl2'].grid(column=1, row=1, **west_grid_params)

        # Buttons are in the mainloop window, not in a Frame.
        self.button['update'].grid(column=0, row=2, **button_grid_params)
        self.button['save_results'].grid(column=0, row=3, **button_grid_params)
        self.button['new_input'].grid( column=0, row=4, **button_grid_params)

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_reqwidth.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Needed across platforms.
        conf_padx = (self.slider['confidence_lbl'].winfo_reqwidth() + 10, 5)
        std_padx = (self.entry['size_std_lbl'].winfo_reqwidth() + 10, 0)

        self.slider['confidence'].grid(column=0, row=0, padx=conf_padx,
                                       columnspan=2, sticky=tk.EW,
                                       )
        self.entry['size_entry'].grid(column=0, row=1, padx=std_padx,)

    def grid_img_labels(self) -> None:
        """
        Grid all image Labels in the dictionary attribute that is defined
        in setup_image_windows(). Called from start_now().
        """

        for lbl in self.img_label:
            self.img_label[lbl].grid(**const.PANEL_LEFT)

    def display_images(self) -> None:
        """
        Ready all image windows for display. Show the processed image in
        its window. Bind rt-click to save any displayed image.
        Called from start_now().

        Returns:
            None
        """

        # Display the input image. It is changed with call_cmd().new_input().
        self.update_image(img_name='input')
        self.update_image(img_name='sized')

        # Now is time to show the sized objects window that was hidden
        #  in setup_image_windows(). Update() speeds up its display.
        self.tkimg_window['sized'].wm_deiconify()
        self.update()


def run_checks() -> None:
    """
    Check system, versions, and command line arguments.
    Program exits if any critical check fails or if the argument
    --about is used, which prints 'about' info, then exits.
    Module check_platform() also enables display scaling on Windows.

    Returns:
            None
    """
    utils.check_platform()
    vcheck.minversion('3.10')
    vcheck.maxversion('3.12')
    manage.arguments()


def main() -> None:
    """
    Main function to launch the program. Initializes SetupApp() and
    sets up the mainloop window and all other windows and widgets.
    Through inheritance, SetupApp() also initializes ProcessImage(),
    which initializes ProcessImage() that inherits Tk, thus creating the
    mainloop window for settings and reporting. With this structure,
    instance attributes and methods are available to all classes
    where needed.
    """

    # Check system, versions, and command line arguments.
    # Exit if any critical check fails or if the --about argument is used.
    # Comment out if using PyInstaller to create an executable.
    #  PyInstaller for Windows will still need to run check_platform()
    #  for DPI Awareness scaling issues.
    run_checks()

    # Instantiating SetupApp() initializes the mainloop window through
     #  multilevel inheritance. The mainloop window originates from
    #  ProcessImage(), which inherits from Tk, thus creating the window.
    app = SetupApp()
    print(f'{PROGRAM_NAME} has launched...')
    app.title(f'{PROGRAM_NAME} Report & Settings')
    app.start_now()

    # Allow user to quit from the Terminal command line using Ctrl-C
    #  without the delay of waiting for tk event actions.
    # Source: https://stackoverflow.com/questions/39840815/
    #   exiting-a-tkinter-app-with-ctrl-c-and-catching-sigint
    # Keep polling the mainloop to check for the SIGINT signal, Ctrl-C.
    # Comment out the following statements before mainloop() when using PyInstaller.
    signal(signalnum=SIGINT,
           handler=lambda x, y: utils.quit_gui(mainloop=app, confirm=False),
           )

    def tk_check(msec):
        app.after(msec, tk_check, msec)

    poll_ms = 500
    app.after(poll_ms, tk_check, poll_ms)

    app.mainloop()

if __name__ == '__main__':
    main()
