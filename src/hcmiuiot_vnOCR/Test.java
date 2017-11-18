package hcmiuiot_vnOCR;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Test {
   public static void main( String[] args ) {
      System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
      //Mat mat = Mat.eye( 3, 3, CvType.CV_8UC1 );
      Mat mat = Imgcodecs.imread("lena.jpg");
      Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
      Imgproc.threshold(mat, mat, 0, 255, Imgproc.THRESH_OTSU);
      Imgcodecs.imwrite("lenaBlackWhite.png", mat);
     
      System.out.println( "mat = " + mat.dump() );
   }
}