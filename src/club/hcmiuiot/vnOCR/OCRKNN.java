package club.hcmiuiot.vnOCR;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.ml.StatModel;

public class OCRKNN {

	private static KNearest knn = KNearest.create();
	private static String trainedFile = "trained.txt";
	
	public OCRKNN() {
		//svm.load(trainedFile);
	}
	
	public OCRKNN(String trainedFile) {
		this.trainedFile = trainedFile;
		new OCRKNN();
	}
	
	public static void train() {
		
		Mat img = new Mat();
		Mat label = new Mat(0,0, CvType.CV_32SC1);

		Mat imgTrain = new Mat(0,0, CvType.CV_32F);
		
		for (int digit=0; digit<10; digit++) {
			for (int element=digit*500; element < (digit+1)*500 - 100; element++) {
				img = Imgcodecs.imread("data/"+digit + "/" + element + ".jpg", 0);
				img.convertTo(img, CvType.CV_32F);
				img = img.clone().reshape(1,1);
				label.push_back(new MatOfInt(digit));
				imgTrain.push_back(img);
			}			
		}

		knn.train(imgTrain, Ml.ROW_SAMPLE, label);
		knn.save(trainedFile);
		
		System.out.println(knn.isTrained());	
	}
	
	private static int predict(String fileName) {	
		Mat a = Imgcodecs.imread(fileName,0);
		Imgproc.resize(a, a, new Size(20,20));
		a.convertTo(a, CvType.CV_32F);
		a = a.clone().reshape(1,1);
		Mat res = new Mat();
		knn.findNearest(a, 10, res);
		//System.out.println("Predicted " + res.dump());
		return (int) res.get(0, 0)[0];
	}
	
	public static void splitAndSave(String fileName) {
		Mat img = Imgcodecs.imread(fileName);
		Mat digit = new Mat();
		
		Rect rec = new Rect(0, 0, 20, 20);
		
		Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
		Imgproc.adaptiveThreshold(img, img, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 5, 0);
		//ImgShow.imshow("digits", img);
		int c = 0;
		
		for (int i=0; i< 50; i++) {
			for (int j=0; j<100; j++) {
				rec.x = 20*j;
				rec.y = 20*i;
				digit = new Mat(img, rec);
				Imgcodecs.imwrite("data/"+(i/5)+"/"+c+".jpg", digit, new MatOfInt(Imgcodecs.CV_IMWRITE_PXM_BINARY));
				//digit.convertTo(digit, );
				System.out.println("> data/"+(i/5)+"/"+c+".jpg");
				//train(digit);
				c++;
			}
		}	
	}
	
	private Mat getSVM() {
		return null;
	}
	
	public int ocr(Mat img) {
		return 0;
	}
	
}
