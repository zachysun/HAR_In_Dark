import cv2
import numpy as np
from PIL import Image


class LowLightEnhancer:
    def __init__(self, video_path=None, enhanced_video_path=None, enhance_methods=['gamma_correction']):
        self.video_path = video_path
        self.enhanced_video_path = enhanced_video_path
        self.enhance_methods = enhance_methods

    def enhance_video(self):
        # initialization and determine whether the file needs to be saved
        is_save = None
        out = None
        if self.enhanced_video_path:
            is_save = True

        capture = cv2.VideoCapture(self.video_path)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        if is_save:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(self.enhanced_video_path, fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

        frames = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            enhanced_frame = self.enhance_frame(frame)
            if is_save:
                out.write(enhanced_frame)
            frames.append(enhanced_frame)

            # for preview
            cv2.imshow('Original Video', frame)
            cv2.imshow('Enhanced Video', enhanced_frame)
            if cv2.waitKey(25) & 0xFF == 27:  # exit
                break

        capture.release()
        if is_save:
            out.release()
        cv2.destroyAllWindows()

        return np.array(frames).transpose((0, 3, 1, 2))

    def enhance_frame(self, frame):
        enhanced_frame = frame
        for method in self.enhance_methods:
            if method == 'hist_equalization':
                enhanced_frame = self._hist_equalization(enhanced_frame)
            elif method == 'clahe':
                enhanced_frame = self._clahe(enhanced_frame)
            elif method == 'gamma_correction':
                enhanced_frame = self._gamma_correction(enhanced_frame)
            elif method == 'gauss_blur':
                enhanced_frame = self._gauss_blur(enhanced_frame)
            elif method == 'bright_contrast_adjust':
                enhanced_frame = self._bright_contrast_adjust(enhanced_frame)

        return enhanced_frame

    # previous: for pytorch transform pipline --> modified: process before pytorch transform pipeline
    def __call__(self, img):
        img_np = np.array(img)
        enhanced_img = self.enhance_frame(img_np)

        return enhanced_img

    @staticmethod
    def _bright_contrast_adjust(img, alpha=1.5, beta=5.0):
        # (c, h, w) to (h, w, c)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return cv2.convertScaleAbs(img, alpha, beta)

    @staticmethod
    def _hist_equalization(img):
        # (c, h, w) to (h, w, c)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_img = cv2.equalizeHist(img_gray)

        return cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _clahe(img, clip_limit=8.0, tile_grid_size=(15, 15)):
        # (c, h, w) to (h, w, c)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
        enhanced_img = clahe.apply(img_gray)

        return cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _gamma_correction(img, gamma=0.4):
        # (c, h, w) to (h, w, c)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        gamma_corrected = (img / 255.0) ** gamma * 255.0

        return gamma_corrected.astype('uint8')

    # mainly smooth noise
    @staticmethod
    def _gauss_blur(img, kernel_size=(5, 5), sigma=.0):
        # (c, h, w) to (h, w, c)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        return cv2.GaussianBlur(img, kernel_size, sigma)


if __name__ == "__main__":
    # ***For Debugging***
    video_path = 'data/validate/244.mp4'
    enhanced_video_path = 'single_enhanced_video/Jump_8_1_enhanced.mp4'
    enhance_methods = ['gamma_correction']
    # enhance video
    img_enhancer = LowLightEnhancer(video_path, None, enhance_methods)
    enhanced_frames = img_enhancer.enhance_video()
    print(enhanced_frames.shape)
