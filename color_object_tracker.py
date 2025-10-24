import cv2
import numpy as np
from collections import deque

class ColorObjectTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.pts = deque(maxlen=64)
        
        # Define color ranges in HSV
        self.color_ranges = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255])
        }
        
        self.current_color = 'blue'
        self.tracking_enabled = True
        self.show_mask = False
        
    def create_trackbars(self):
        """Create trackbars for HSV adjustment"""
        cv2.namedWindow('HSV Adjustments')
        cv2.createTrackbar('H Lower', 'HSV Adjustments', 100, 179, lambda x: None)
        cv2.createTrackbar('H Upper', 'HSV Adjustments', 130, 179, lambda x: None)
        cv2.createTrackbar('S Lower', 'HSV Adjustments', 50, 255, lambda x: None)
        cv2.createTrackbar('S Upper', 'HSV Adjustments', 255, 255, lambda x: None)
        cv2.createTrackbar('V Lower', 'HSV Adjustments', 50, 255, lambda x: None)
        cv2.createTrackbar('V Upper', 'HSV Adjustments', 255, 255, lambda x: None)
        
    def get_hsv_from_trackbars(self):
        """Get current HSV values from trackbars"""
        h_lower = cv2.getTrackbarPos('H Lower', 'HSV Adjustments')
        h_upper = cv2.getTrackbarPos('H Upper', 'HSV Adjustments')
        s_lower = cv2.getTrackbarPos('S Lower', 'HSV Adjustments')
        s_upper = cv2.getTrackbarPos('S Upper', 'HSV Adjustments')
        v_lower = cv2.getTrackbarPos('V Lower', 'HSV Adjustments')
        v_upper = cv2.getTrackbarPos('V Upper', 'HSV Adjustments')
        
        return ([h_lower, s_lower, v_lower], [h_upper, s_upper, v_upper])
    
    def detect_and_track(self, frame):
        """Main detection and tracking logic"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range
        lower, upper = self.get_hsv_from_trackbars()
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(c) > 500:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(c)
                
                # Calculate moments for centroid
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Draw contour and bounding box
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                    # Display information
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    
                    cv2.putText(frame, f"Area: {int(area)}", (x, y - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Perimeter: {int(perimeter)}", (x, y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Center: {center}", (x, y - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update tracking points
        if self.tracking_enabled and center is not None:
            self.pts.appendleft(center)
        
        # Draw tracking trail
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 255, 255), thickness)
        
        return frame, mask
    
    def run(self):
        """Main application loop"""
        self.create_trackbars()
        
        print("=== Color Object Tracker ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle tracking trail")
        print("  'm' - Toggle mask view")
        print("  'c' - Clear tracking points")
        print("  'r/g/b/y' - Switch to Red/Green/Blue/Yellow preset")
        print("\nAdjust HSV sliders to fine-tune detection")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror view
            
            # Detect and track objects
            result_frame, mask = self.detect_and_track(frame)
            
            # Add instructions overlay
            cv2.putText(result_frame, f"Tracking: {self.current_color.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, "Press 'h' for help", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frames
            cv2.imshow('Color Object Tracker', result_frame)
            
            if self.show_mask:
                cv2.imshow('Mask', mask)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.tracking_enabled = not self.tracking_enabled
                print(f"Tracking trail: {'ON' if self.tracking_enabled else 'OFF'}")
            elif key == ord('m'):
                self.show_mask = not self.show_mask
                if not self.show_mask:
                    cv2.destroyWindow('Mask')
            elif key == ord('c'):
                self.pts.clear()
                print("Tracking points cleared")
            elif key == ord('r'):
                self.current_color = 'red'
                self.apply_color_preset()
            elif key == ord('g'):
                self.current_color = 'green'
                self.apply_color_preset()
            elif key == ord('b'):
                self.current_color = 'blue'
                self.apply_color_preset()
            elif key == ord('y'):
                self.current_color = 'yellow'
                self.apply_color_preset()
            elif key == ord('h'):
                self.show_help()
        
        self.cleanup()
    
    def apply_color_preset(self):
        """Apply preset color range to trackbars"""
        lower, upper = self.color_ranges[self.current_color]
        cv2.setTrackbarPos('H Lower', 'HSV Adjustments', lower[0])
        cv2.setTrackbarPos('H Upper', 'HSV Adjustments', upper[0])
        cv2.setTrackbarPos('S Lower', 'HSV Adjustments', lower[1])
        cv2.setTrackbarPos('S Upper', 'HSV Adjustments', upper[1])
        cv2.setTrackbarPos('V Lower', 'HSV Adjustments', lower[2])
        cv2.setTrackbarPos('V Upper', 'HSV Adjustments', upper[2])
        print(f"Switched to {self.current_color} detection")
    
    def show_help(self):
        """Display help information"""
        print("\n=== HELP ===")
        print("Controls:")
        print("  'q' - Quit application")
        print("  't' - Toggle tracking trail on/off")
        print("  'm' - Toggle mask view")
        print("  'c' - Clear tracking points")
        print("  'r' - Switch to Red detection preset")
        print("  'g' - Switch to Green detection preset")
        print("  'b' - Switch to Blue detection preset")
        print("  'y' - Switch to Yellow detection preset")
        print("\nUse HSV sliders to fine-tune color detection")
        print("============\n")
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ColorObjectTracker()
    tracker.run()