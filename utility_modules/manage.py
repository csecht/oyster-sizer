"""
General housekeeping utilities.
Functions:
arguments: handles command line arguments
input_metrics: reads specified input image and derives associated metrics.
scale: manages the specified scale factor for display of images.
tk_image: converts scaled cv2 image to a compatible tk.TK image format.
ttk_styles: configures platform-specific ttk.Style for Buttons and Comboboxes.
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License

# Standard library imports.
import argparse
import tkinter as tk
from sys import exit as sys_exit
from tkinter import ttk

# Third party imports.
import cv2
import numpy as np
import sys
from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage

# Local application imports.
import utility_modules
from utility_modules import constants as const

def arguments() -> dict:
    """
    Handle command line arguments.
    Returns:
        dictionary of arguments; keys: 'about', 'terminal'.
    """

    parser = argparse.ArgumentParser(description='Image Processing to Size Objects.')
    parser.add_argument('--about',
                        help='Provide description, version, GNU license',
                        action='store_true',
                        default=False)
    parser.add_argument('--terminal', '-t',
                        help='Prints to Terminal the report that is also saved to file.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    about_text = (f'{sys.modules["__main__"].__doc__}\n'
                  f'{"Author:".ljust(13)}{utility_modules.__author__}\n'
                  f'{"Version:".ljust(13)}{utility_modules.__version__}\n'
                  f'{"Status:".ljust(13)}{utility_modules.__status__}\n'
                  f'{"URL:".ljust(13)}{utility_modules.URL}\n'
                  f'{utility_modules.__copyright__}'
                  f'{utility_modules.__license__}\n'
                  )

    if args.about:
        print('====================== ABOUT START ====================')
        print(about_text)
        print('====================== ABOUT END ====================')
        sys_exit(0)

    # The 'about' key is called from setup_start_window() Help menu.
    #  The 'terminal' key is called from utils.save_report_and_img().
    arg_dict = {
        'about': about_text,
        'terminal': args.terminal,
    }

    return arg_dict


def input_metrics(img: np.ndarray) -> dict:
    """
    Read input file, then calculate and assign to a dictionary values
    that can be used as constants for image processing and display.

    Returns:
        Dictionary of image values and metrics; keys:
         'font_scale', 'line_thickness'.
    """

    # h, w, _ = img.shape # or img.shape[1::-1] -> (width, height)
    img_avg_dimension: int = round((img.shape[0] + img.shape[1]) / 2)

    line_thickness: int = max(round(img_avg_dimension * const.LINE_THICKNESS_FACTOR), 1)

    # Ideas for scaling: https://stackoverflow.com/questions/52846474/
    #   how-to-resize-text-for-cv2-puttext-according-to-the-image-object_size-in-opencv-python
    font_scale: float = round(max(img_avg_dimension * const.FONT_SCALE_FACTOR, 0.33), 2)

    metrics = {
        'line_thickness': line_thickness,
        'font_scale': font_scale,
    }

    return metrics


def tk_image(image: np.ndarray, scale_factor: float) -> PhotoImage:
    """
    Scales and converts cv2 images to a compatible format for display
    in tk window. Be sure that the returned image is properly scoped in
    the Class where it is called; e.g., use as self.tk_image attribute.

    Args:
        image: A cv2 numpy array of the image to scale and convert to
               a PIL ImageTk.PhotoImage.
        scale_factor: A float number; from user selection or an
            auto-scale statement.

    Returns:
        Scaled PIL ImageTk.PhotoImage for display as a tk.Label image arg.
    """

    # Need to scale images for display; images for processing are left raw.
    scale_factor = 1 if scale_factor == 0 else scale_factor

    # Provide the best interpolation method for slight improvement of
    #  resized image depending on whether it is down- or up-scaled.
    interpolate = cv2.INTER_AREA if scale_factor < 0 else cv2.INTER_CUBIC

    scaled_img = cv2.resize(src=image,
                            dsize=None,
                            fx=scale_factor, fy=scale_factor,
                            interpolation=interpolate)

    # based on tutorial: https://pyimagesearch.com/2016/05/23/opencv-with-tkinter/
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)

    scaled_img = Image.fromarray(scaled_img)
    tk_img: PhotoImage = ImageTk.PhotoImage(scaled_img)
    # Need to prevent garbage collection to show image in tk.Label, etc.
    tk_img.image = tk_img

    return tk_img


def ttk_styles(mainloop: tk.Tk) -> None:
    """
    Configure platform-specific ttk.Style for Buttons and Comboboxes.
    Font and color values need to be edited as appropriate for the
    application (to avoid lengthy parameter arguments).

    Args:
         mainloop: The tk.Toplevel running as the mainloop.

    Returns:
        None
    """

    ttk.Style().theme_use('alt')

    # Use fancy buttons for Linux and Windows, standard theme for macOS,
    #   but with a custom font.
    bstyle = ttk.Style()
    combo_style = ttk.Style()

    if const.MY_OS == 'lin':
        font_size = 8
    elif const.MY_OS == 'win':
        font_size = 7
    else:  # is macOS
        font_size = 9

    bstyle.configure("My.TButton", font=('TkTooltipFont', font_size))
    mainloop.option_add("*TCombobox*Font", ('TkTooltipFont', font_size))

    if const.MY_OS == 'lin':
        bstyle.map("My.TButton",
                   foreground=[('active', const.COLORS_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.COLORS_TK['vermilion'])],
                   )
        combo_style.map('TCombobox',
                        fieldbackground=[('readonly',
                                          const.COLORS_TK['dark blue'])],
                        selectbackground=[('readonly',
                                           const.COLORS_TK['dark blue'])],
                        selectforeround=[('readonly',
                                          const.COLORS_TK['yellow'])],
                        )
    elif const.MY_OS == 'win':
        bstyle.map("My.TButton",
                   foreground=[('active', const.COLORS_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.COLORS_TK['vermilion'])],
                   )
