import cv2
'''
Drawing a line on an image
• cv2.line(image, (x1, y1), (x2, y2), (color), thickness)
• (x1, y1) are the starting coordinates of the line
• (x2, y2) are the ending coordinates of the line
• Color is the color of the line to be drawn in RGB
• Thickness is the thickness of the line in px
'''
def draw_line(image, start_point, end_point, color, thickness):
    return cv2.line(image, start_point, end_point, color, thickness)
'''
Drawing a rectangle:
• cv2.rectangle (image, (x1, y1), (x2, y2), (color), thickness)
• (x1, y1) is the starting coordinate
• (x2, y2) is the ending coordinate
• (color) is color in RGB
• Thickness in thickness in px
'''
def draw_rectangle(image, start_point, end_point, color, thickness):
    return cv2.rectangle(image, start_point, end_point, color, thickness)
'''
Drawing a circle:
• cv2.circle (image, (center x, center y), radius, (color), thickness)
• (center x, center y) is the center coordinate of the circle

'''
def draw_circle(image, center, radius, color, thickness):
    return cv2.circle(image, center, radius, color, thickness)
'''
Adding a text to an image
• cv2.putText(image, ‘text’, (x, y), font, fontScale, (color), thickness)
• (x, y) is the coordinate of the bottom-left corner of the text string in the image
• Font denotes the font type. (e.g. FONT_HERSHEY_SIMPLEX,
FONT_HERSHEY_PLAIN)
• fontScale is a font scale factor that is multiplied by the font specific base size
• Color is the color of text string to be drawn in RGB
• Thickness if the thickness of the line in px
https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

'''
def add_text(image, text, position, font, font_scale, color, thickness):
    return cv2.putText(image, text, position, font, font_scale, color, thickness)