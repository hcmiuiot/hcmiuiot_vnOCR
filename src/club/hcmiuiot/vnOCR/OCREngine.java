package club.hcmiuiot.vnOCR;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import club.hcmiuiot.opencv.ImgShow;

public class OCREngine {	
	
	public OCREngine() {
		
		System.loadLibrary("opencv_hcmiuiot");
		
	}
	
	
	public static void main(String[] args) {
		new OCREngine();
		OCRKNN.train();
		String fileName = "data/testData/test16.jpg";
		
		Mat img = Imgcodecs.imread(fileName, 0);
		Mat src = Imgcodecs.imread(fileName);
		
		Imgproc.threshold(img, img, 100, 255, Imgproc.THRESH_BINARY);
		
		ImgShow.imshow("src", img);
		
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Mat roi = new Mat();
		
		Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		
		for (int i=0; i<contours.size(); i++) {
			Rect rec = Imgproc.boundingRect(contours.get(i));
			roi = img.submat(rec);
			
			int predictValue = OCRKNN.predict(roi);
			
			Imgproc.rectangle(src, rec.tl(), rec.br(), new Scalar(255,0,0),2);
			Imgproc.putText(src, ""+predictValue, new Point(rec.tl().x, rec.tl().y), 1, 2f, new Scalar(0,255,0));
		}
		
		ImgShow.imshow("src", src);
	}
	
}
