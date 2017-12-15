package club.hcmiuiot.vnOCR;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import club.hcmiuiot.opencv.ImgShow;

public class OCREngine {	
	
	public OCREngine() {
		
		System.loadLibrary("opencv_hcmiuiot");
		
//		Mat img = Imgcodecs.imread("data/digits.png");
//		ImgShow.imshow("src", img);
//			
//		HOGDescriptor hog = new HOGDescriptor(new Size(20,20),
//												new Size(10,10), 
//												new Size(5,5), 
//												new Size(10,10), 
//												9);
//		
//		
//		
//		MatOfFloat descriptors = new MatOfFloat();
//		hog.compute(img, descriptors);
//		
		//System.out.println(descriptors.dump());
		
	}
	
	
	public static void main(String[] args) {
		new OCREngine();
		//OCRSVM.splitAndSave("data/digits.png");
//		Mat m = Imgcodecs.imread("data/0/0.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
//		System.out.println(m.dump());
		OCRKNN.train();
		System.out.println("test.jpg");
	}
	
}
