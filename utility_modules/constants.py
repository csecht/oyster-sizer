"""
Constants used throughout main script and modules:
MY_OS
MODEL_NAME
PREDICT_IMGSZ
PREDICT_CONF
PREDICT_IOU
PREDICT_MAX_DET
PREDICT_HALF
PREDICT_AUGMENT
EDGE_PROXIMITY
TIME_STAMP_FORMAT
TIME_PRINT_FORMAT
STUB_ARRAY
FONT_SCALE_FACTOR
LINE_THICKNESS_FACTOR
COLORS_CV
COLORS_TK
FONT_TYPE
OS_SETTINGS
settings
C_KEY
C_BIND
REPORT_FONT
WIDGET_FONT
MENU_FONT
TIPS_FONT
MASTER_BG
DARK_BG
DRAG_GRAY
WIDGET_FG
LABEL_PARAMETERS
SCALE_PARAMETERS
WINDOW_PARAMETERS
COMBO_PARAMETERS
PANEL_LEFT
PANEL_RIGHT
WINDOW_TITLES
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License'

# Standard library import
from sys import platform

# Third party import
import cv2
import numpy as np

MY_OS: str = platform[:3]  # 'lin', 'win', or 'dar'

# The YOLO model and Ultralytics prediction() function arguments.
MODEL_NAME = 'oyster_yolo11n_960_103e_20b'
PREDICT_IMGSZ = 960  # set to match model training size, default 640. Must be multiple of max stride 32.
PREDICT_IOU = 0.70  # intersection-over-union threshold, default 0.70
PREDICT_MAX_DET = 400  # maximum detections per image, default 300
# When possible, use half precision to speed prediction; default False.
PREDICT_HALF = False if MY_OS == 'dar' else True
PREDICT_AUGMENT = False  # augment images for prediction, default False

# Minimum px distance between "interior" object and img edge.
EDGE_PROXIMITY = 3

# 1.1 is a more conservative threshold for smaller oysters.
# 1.17 bbox_ratio_mean is a rough threshold for mature oysters, from observations.
BOX_RATIO_THRESHOLD = 1.15 # 1.17

# Dictionary used to adjust the correction factor based on the box ratio.
# Values empirically determined based on manual calibration measurements of
#   mature oysters. Needs verification with younger oysters.
#  Used to create linear equation in ViewImage.get_correction_factor().
# Keys are the bbox_ratio_mean, values are the correction factor.
correction_factors = {
    1.15: 1.005,
    1.16: 1.01,
    1.17: 1.015,
    1.18: 1.02,
    1.19: 1.025,
    1.20: 1.03,
    1.21: 1.035,
    1.22: 1.04,
    1.23: 1.045,
    1.24: 1.05,
    1.25: 1.055,
    1.26: 1.06,
    1.27: 1.065,
    1.28: 1.07,
    1.29: 1.075,
    1.30: 1.08,
    1.31: 1.085,
    1.32: 1.09,
    1.33: 1.095,
    1.34: 1.10,
    1.35: 1.105,
    1.36: 1.11,
}

TIME_STAMP_FORMAT = '%Y%m%d%I%M%S'  # for file names, as: 20240301095308
TIME_PRINT_FORMAT = '%c'  # as: Fri Mar 1 09:53:08 2024, is locale-dependent.
# TIME_PRINT_FORMAT = '%Y-%m-%d %I:%M:%S %p'  # as: 2024-03-01 09:53:08 AM

# The stub is a white square set in a black square to make it obvious when
#  the code has a bug.
STUB_ARRAY: np.ndarray = cv2.rectangle(np.zeros(shape=(200, 200), dtype="uint8"),
                                       (50, 50), (150, 150), 255, -1)

# Scaling factors for text and lines; empirically determined.
#  Used in manage.py input_metrics().
FONT_SCALE_FACTOR: float = 5.5e-4
LINE_THICKNESS_FACTOR: float = 1.5e-3

# Use for cv2.addWeighted transparent text box.
ALPHA = 0.5

# Colorblind color pallet source:
#   Wong, B. Points of view: Color blindness. Nat Methods 8, 441 (2011).
#   https://doi.org/10.1038/nmeth.1618
# See also: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# OpenCV uses a BGR (B, G, R) color convention, instead of RGB.
COLORS_CV = {
    'blue': (178, 114, 0),
    'orange': (0, 159, 230),
    'dark blue': (112, 25, 25),
    'sky blue': (233, 180, 86),
    'blueish green': (115, 158, 0),
    'vermilion': (0, 94, 213),
    'reddish purple': (167, 121, 204),
    'yellow': (66, 228, 240),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'DarkOrchid1': (255, 62, 191),
    'gold1': (0, 215, 255),
}

# Set 'tk_white' based on the operating system's default white.
# This structure directly maps the operating system to the corresponding
#  color for 'tk_white'. This eliminates the need for a if-elif-else
#  structure. The get method for 'tk_white' is used to provide a default
#  value of 'grey95' (Windows) if the operating system is not 'dar' (macOS)
#  or 'lin' (Linux).
COLORS_TK = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'dark blue': 'MidnightBlue',
    'sky blue': '#56B4E9',
    'blueish green': '#009E73',
    'vermilion': '#D55E00',
    'reddish purple': '#CC79A7',
    'yellow': '#F0E442',
    'black': 'black',
    'white': 'white',
    'tk_white': {'dar': 'white', 'lin': 'grey85'}.get(MY_OS, 'grey95'),
    'red': 'red1',  # not compatible for good color-blindcontrast
    'green': 'green1',  # not compatible for good color-blind contrast
}

# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
# 	cv::HersheyFonts {
#   cv::FONT_HERSHEY_SIMPLEX = 0, # cv2 default
#   cv::FONT_HERSHEY_PLAIN = 1,
#   cv::FONT_HERSHEY_DUPLEX = 2,
#   cv::FONT_HERSHEY_COMPLEX = 3,
#   cv::FONT_HERSHEY_TRIPLEX = 4,
#   cv::FONT_HERSHEY_COMPLEX_SMALL = 5,
#   cv::FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
#   cv::FONT_HERSHEY_SCRIPT_COMPLEX = 7,
#   cv::FONT_ITALIC = 16
# }
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

OS_SETTINGS = {
    'lin': {
        'os_font': 'DejaVu Sans',
        'os_mono_font': 'DejaVu Sans Mono',
        'widget_font_size': (8,),
        'report_font_size': (9,),
        'menu_font_size': (9,),
        'tips_font_size': (8,),
        'radio_params': {
            'fg': COLORS_TK['yellow'],
            'activebackground': 'gray50',
            'activeforeground': COLORS_TK['sky blue'],
            'selectcolor': COLORS_TK['dark blue'],
        },
        'c_key': 'Ctrl',
        'c_bind': 'Control',
    },
    'win': {
        'os_font': 'Segoe UI',
        'os_mono_font': 'Consolas',
        'widget_font_size': (7,),
        'report_font_size': (8,),
        'menu_font_size': (9,),
        'tips_font_size': (8,),
        'radio_params': {'fg': 'black'},
        'c_key': 'Ctrl',
        'c_bind': 'Control',
    },
    'dar': {
        'os_font': 'SF Pro',
        'os_mono_font': 'Menlo',
        'widget_font_size': (10,),
        'report_font_size': (10,),
        'menu_font_size': (13,),
        'tips_font_size': (11,),
        'radio_params': {'fg': 'black'},
        'c_key': 'Command',
        'c_bind': 'Command',
    }
}

# Defaults to Windows if OS is not 'lin' or 'dar'.
settings = OS_SETTINGS.get(MY_OS, OS_SETTINGS['win'])

C_KEY = settings['c_key']
C_BIND = settings['c_bind']

REPORT_FONT = settings['os_mono_font'], *settings['report_font_size']
WIDGET_FONT = settings['os_font'], *settings['widget_font_size']
MENU_FONT = settings['os_font'], *settings['menu_font_size']
TIPS_FONT = settings['os_font'], *settings['tips_font_size']

MASTER_BG = COLORS_TK['white']  # or COLORS_TK['tk_white'] for off-white.
DARK_BG = 'gray20'
DRAG_GRAY = 'gray65'
WIDGET_FG = COLORS_TK['yellow']

LABEL_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg=DARK_BG,
    fg=WIDGET_FG,
)

SCALE_PARAMETERS = dict(
    width=8,
    orient='horizontal',
    showvalue=False,
    sliderlength=18,
    font=WIDGET_FONT,
    bg=COLORS_TK['dark blue'],
    fg=WIDGET_FG,
    troughcolor=MASTER_BG,
)

# Color-in the main (self) window and give it a yellow border;
#  border highlightcolor changes to grey with loss of focus.
WINDOW_PARAMETERS = dict(
    bg=DARK_BG,
    # bg=COLORS_TK['sky blue'],  # for development
    highlightthickness=5,
    highlightcolor=COLORS_TK['yellow'],
    highlightbackground=DRAG_GRAY,
    padx=3, pady=3,
)

# Grid arguments to position tk.Label images in their windows.
PANEL_LEFT = dict(
    column=0, row=0,
    padx=5, pady=5,
    sticky='w',
)

PANEL_RIGHT = dict(
    column=1, row=0,
    padx=5, pady=5,
    sticky='e',
)

# Item order determines initial window layer order, first on top.
WINDOW_TITLES = {
    'sized': 'Sized Objects',
    'input': 'Input image',
}

# Values are in mm units.
# Value of 1.001 for 'None' is a hack to force 4 sig.fig as the default.
#  This allows the most accurate display of pixel widths at startup,
#  assuming that object_type sizes are limited to <10,000 pixel diameters.
# SIZE_STANDARDS = {
#     'None': 1.001,
#     'Custom': 0,
#     'Puck': 76.2,
#     'Cent': 19.0,
#     'Nickel': 21.2,
#     'Dime': 17.9,
#     'Quarter': 24.3,
#     'Half Dollar': 30.6,
#     'Sacagawea $': 26.5,
#     'Eisenhower $': 38.1
# }
