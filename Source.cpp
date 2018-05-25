#define _USE_MATH_DEFINES
#include <cmath>


#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;



// this class is for OpenCV ParallelLoopBody
class Parallel_pixel_SetToZero : public ParallelLoopBody
{
private:
	uchar *p ;
public:
	Parallel_pixel_SetToZero(uchar* ptr ) : p(ptr) {}

	virtual void operator()( const Range &r ) const
	{
		for ( register int i = r.start; i != r.end; ++i)
		{
			if(p[i]<50)
				p[i]=0;
		}
	}
};


Mat EdgeEnhancemen(Mat src_gray)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad;
	Mat nr_grad;
	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	//--- standard horizontal and vertical Sobel 
	// Gradient X
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	grad_x.release();

	// Gradient Y
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	grad_y.release();
	//-

	// gradient magnitude image is obtained by summing two resultant images
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	abs_grad_x.release();
	abs_grad_y.release();

	// normalization is applied to stretch the pixel values in the whole range [0–255]
	normalize(grad,nr_grad,0,255,NORM_MINMAX);
	Mat new1= grad.clone();
	Mat new2= nr_grad.clone();
	return nr_grad;
}


double PowerOfRegion(Mat frame_gray)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat frame_gray_guss=frame_gray.clone();
	/// reduce noise
	//GaussianBlur( frame_gray, frame_gray_guss, Size(3,3), 0, 0, BORDER_DEFAULT );


	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel( frame_gray_guss, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	Sobel( frame_gray_guss, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	int NR=frame_gray_guss.rows*frame_gray_guss.cols;

	//--- set the gray-levels of less than 50 to zero 
	uchar* p3 = abs_grad_x.data ;
	parallel_for_( Range(0,NR) , Parallel_pixel_SetToZero(p3)) ;

	uchar* p4 = abs_grad_y.data ;
	parallel_for_( Range(0,NR) , Parallel_pixel_SetToZero(p4)) ;
	//--

	float PR;
	Mat pow2_grad_x,pow2_grad_y;

	pow(abs_grad_x,2.0,pow2_grad_x);
	pow(abs_grad_y,2.0,pow2_grad_y);

	double nr=((double)1/(double)NR);
	cout<<"nr"<<nr<<endl;
	PR=nr*sum(pow2_grad_x).val[0]+nr*sum(pow2_grad_y).val[0];

	return PR;
}


Rect chooseSearchWindow(Mat frame, Rect templateLoc)
{
	int w=floor(templateLoc.width/2)+10,
		h=floor(templateLoc.height/2)+10;

	int X=templateLoc.x-w,	W=templateLoc.width+2*w,
		Y=templateLoc.y-h,	H=templateLoc.height+2*h,
		diffX=0,
		diffY=0;
	if(X<0)
	{
		diffX=abs(X);
		X=0;
	}
	if(Y<0)
	{
		diffY=abs(Y);
		Y=0;
	}
	W+=diffX;
	H+=diffY;
	if(X+W>frame.cols)
		W=frame.cols-X;
	if(Y+H>frame.rows)
		H=frame.rows-Y;
	return Rect(Point(X,Y),Point(X+W,Y+H));
}
float roundf(float x)
{
	return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}
float roundMax( float val)
{
	return roundf(val * 1000) / 1000;
}



template <typename T>T **AllocateDynamicArray( int nRows, int nCols)
{
	T **dynamicArray;

	dynamicArray = new T*[nRows];
	for( int i = 0 ; i < nRows ; i++ )
		dynamicArray[i] = new T [nCols];

	return dynamicArray;
}

template <typename T>
void FreeDynamicArray(T** dArray)
{
	delete [] *dArray;
	delete [] dArray;
}






//---------------------------------------------------------------------------------------------------------
/** Global variables */
String cascade_name = "airplanes22stage.xml";
CascadeClassifier cascade;

int ii=0;
/** @function main */
int main()
{
	VideoCapture capture;
	Mat frame;

	//-- Load the cascades
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

	//-- Read the video stream
	std::string video_name="Airplane2.mp4";
	capture.open( video_name);

	//-- exit if Error opening video capture
	if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

	bool isMissed=true;//in start we don't have any template in frame
	vector<Rect> templateLoc;//Location of template as rectangle
	vector<Mat> _template;//template as Mat(image)
	vector<Rect> searchWin;//search area for find template = search window

	while (  capture.read(frame) )
	{
		if( frame.empty() )
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		Mat frame_gray;
		cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

		if(isMissed)
		{
			//-- Detect airplanes
			cascade.detectMultiScale( frame_gray, templateLoc, 1.1, 2,0|CASCADE_SCALE_IMAGE, Size(1, 1) );
			
			if(templateLoc.size()!=0)
			{
				for (int k = 0; k < templateLoc.size(); k++)
				{
					//-- convert template - rect to Mat 
					_template.push_back(frame_gray(templateLoc[k]));
					//-- choose search window for find template in limit area
					searchWin.push_back(chooseSearchWindow( frame,  templateLoc[k]));
				}
				isMissed=false;// now we have template
			}
			//else //can't find any template in frame
		}
		else
		{
			//---------------------------------------------------------------------------------
			//---------------------------------------------------------------------------------
			//-----------------------------STEP 1----------------------------------------------
			for (int k = 0; k < templateLoc.size(); k++)
			{

				bool isCrowded=false;
				int flag=0;
				float T1=70;//intensity thershold
				float T2=0.15;//relative power thershold
				float T3=0.84;
				int w=15;// floor(1/4*templateLoc.width);
				int h=w;// floor(1/4*templateLoc.height);

				//calculate intensity measure --------------------------------------------
				bool is4RegionInFrame=false;
				if(templateLoc[k].y-h>=1 
					&& templateLoc[k].x-w>=1
					&& templateLoc[k].y+templateLoc[k].height+w<frame_gray.rows
					&& templateLoc[k].x+templateLoc[k].width+w<frame_gray.cols)
				{
					is4RegionInFrame=true;
				}

				if(is4RegionInFrame)
				{
					Rect up;
					up = Rect(Point(templateLoc[k].x,templateLoc[k].y-h),Point(templateLoc[k].x+templateLoc[k].width,templateLoc[k].y));
					rectangle( frame, up, Scalar( 0, 0, 255 ));

					Rect left;
					left=Rect(Point(templateLoc[k].x-w,templateLoc[k].y),Point(templateLoc[k].x,templateLoc[k].y+templateLoc[k].height));
					rectangle( frame, left, Scalar( 0, 0, 255 ));		

					Rect bottum(Point(templateLoc[k].x,templateLoc[k].y+templateLoc[k].height),Point(templateLoc[k].x+templateLoc[k].width,templateLoc[k].y+templateLoc[k].height+w));
					rectangle( frame, bottum, Scalar( 0, 0, 255 ));		

					Rect right(Point(templateLoc[k].x+templateLoc[k].width,templateLoc[k].y),Point(templateLoc[k].x+templateLoc[k].width+w,templateLoc[k].y+templateLoc[k].height));
					rectangle( frame, right, Scalar( 0, 0, 255 ));

					//-- display search window
					//rectangle( frame, searchWin, Scalar( 0, 255, 0 ));

					Mat _up= frame_gray( up );
					Mat _left= frame_gray( left );
					Mat _bottum= frame_gray( bottum );
					Mat _right= frame_gray( right );

					Mat _UB;
					absdiff(_up,_bottum,_UB);
					float UB=mean(_UB).val[0];

					Mat _LR;
					absdiff(_left,_right,_LR);
					float LR= mean(_LR).val[0];
					//end - calculate intensity measure --------------------------------------------


					if(UB>T1 || LR>T1) // intensity
					{	
						cout<<UB<<"\t";
						cout<<LR<<"\t"<<"crowded"<<endl;
						isCrowded=true;
						//cout<<isCrowded<<"\t"<<ii<<endl;
					}
					else // calculate relative power
					{
						float PR_u=PowerOfRegion(_up);
						float PR_l=PowerOfRegion(_left);
						float PR_b=PowerOfRegion(_bottum);
						float PR_r=PowerOfRegion(_right);
						float PR_t=PowerOfRegion(_template[k]);
						cout<<PR_u<<"\t"<<PR_l<<"\t"<<PR_b<<"\t"<<PR_r<<"\t"<<PR_t<<endl;

						if((PR_u/PR_t)>T2 ||(PR_l/PR_t)>T2 ||(PR_b/PR_t)>T2 ||(PR_r/PR_t)>T2)
						{
							cout<<"relative power detected"<<"\t"<<"crowded"<<endl;
							isCrowded=true;
						}
					}
				}
				else
				{
					cout<<"rect out of frames"<<endl;
				}

				//---------------------------------------------------------------------------------
				//---------------------------------------------------------------------------------
				//-----------------------------STEP 2----------------------------------------------
				if(isCrowded) // RMI method
				{
					Mat srw=frame_gray(searchWin[k]);
					int MN=(_template[k].cols-2)*(_template[k].rows-2);
					const int colsSW=(srw.cols- _template[k].cols+1);
					const int rowsSW=(srw.rows- _template[k].rows+1);
					//double **RMI = AllocateDynamicArray<double>(colsSW,rowsSW);
					double minRMI=100000000;
					int posX,posY;
					int step=15;
					for(int i2=0;i2<colsSW;i2+=step)
					{
						for(int j2=0;j2<rowsSW;j2+=step)
						{ 
							Rect tLoc(i2,j2,_template[k].cols,_template[k].rows);
							Mat corresponding=srw(tLoc);
							Mat Vect=Mat::zeros(18,MN,CV_8U);
							int ctr=0;
							for(int i3=1;i3<corresponding.cols-1;i3++)
							{
								for (int j3 =1; j3 < corresponding.rows-1; j3++)
								{
									//9 pixel of template
									Vect.at<uchar>(0,ctr)=_template[k].at<uchar>(i3-1,j3-1);
									Vect.at<uchar>(1,ctr)=_template[k].at<uchar>(i3,j3-1);
									Vect.at<uchar>(2,ctr)=_template[k].at<uchar>(i3+1,j3-1);

									Vect.at<uchar>(3,ctr)=_template[k].at<uchar>(i3-1,j3);
									Vect.at<uchar>(4,ctr)=_template[k].at<uchar>(i3,j3);
									Vect.at<uchar>(5,ctr)=_template[k].at<uchar>(i3+1,j3);

									Vect.at<uchar>(6,ctr)=_template[k].at<uchar>(i3-1,j3+1);
									Vect.at<uchar>(7,ctr)=_template[k].at<uchar>(i3,j3+1);
									Vect.at<uchar>(8,ctr)=_template[k].at<uchar>(i3+1,j3+1);

									//9 pixel of corresponding
									Vect.at<uchar>(9,ctr)=corresponding.at<uchar>(i3-1,j3-1);
									Vect.at<uchar>(10,ctr)=corresponding.at<uchar>(i3,j3-1);
									Vect.at<uchar>(11,ctr)=corresponding.at<uchar>(i3+1,j3-1);

									Vect.at<uchar>(12,ctr)=corresponding.at<uchar>(i3-1,j3);
									Vect.at<uchar>(13,ctr)=corresponding.at<uchar>(i3,j3);
									Vect.at<uchar>(14,ctr)=corresponding.at<uchar>(i3+1,j3);

									Vect.at<uchar>(15,ctr)=corresponding.at<uchar>(i3-1,j3+1);
									Vect.at<uchar>(16,ctr)=corresponding.at<uchar>(i3,j3+1);
									Vect.at<uchar>(17,ctr)=corresponding.at<uchar>(i3+1,j3+1);

									ctr++;
								}
							}

							Mat meanVect;
							reduce(Vect,meanVect,1,CV_REDUCE_AVG);

							Mat meanVect2;

							repeat(meanVect,1,MN,meanVect2);

							Mat resVect;
							subtract(Vect,meanVect2,resVect);
							Mat resVect2;
							mulTransposed(resVect,resVect2,0,noArray(),1,-1);
							Mat C;
							double mn= (double)1/(double)MN;
							C=resVect2*(mn);

							double HC=( std::pow((2*M_PI*M_E),0.5) )*( std::pow(determinant(C),0.5) );
							double HCA=( std::pow((2*M_PI*M_E),0.5) )*( std::pow(determinant(C(Rect(0,0,8,8))),0.5) );
							double HCB=( std::pow((2*M_PI*M_E),0.5) )*( std::pow(determinant(C(Rect(9,9,9,9))),0.5) );

							double RMI=(HCA+HCB-HC);
							if(RMI<minRMI)
							{
								minRMI=RMI;
								posX=i2;
								posY=j2;
							}
						}
					}


					//for(int i2=0;i2<colsSW;i2++)
					//{
					//	for(int j2=0;j2<rowsSW;j2++)
					//	{
					//		if(RMI[i2][j2]<minRMI)
					//		{
					//			maxRMI=RMI[i2][j2];
					//			posX=i2;
					//			posY=j2;
					//		}

					//	}
					//}
					Rect tLocInSW(posX+searchWin[k].x, 
						posY+searchWin[k].y, 
						_template[k].cols,
						_template[k].rows);
					templateLoc[k] = Rect(tLocInSW);
					_template[k] = frame_gray(tLocInSW);
					//--- choose search window
					searchWin[k] = chooseSearchWindow( frame,  templateLoc[k]);


					/// Show me what you got
					rectangle( frame, templateLoc[k], Scalar::all(0), 2, 8, 0 );

					//	
				}
				else // EC method-----------------------------------------------------------
				{
					if(ii==85)
						cout<<endl;
					Mat t90,t100,t110;
					t100=_template[k].clone();
					resize(t100, t110, Size(), 1.1, 1.1, CV_INTER_AREA );
					resize(t100, t90, Size(), 0.5, 0.5, CV_INTER_LINEAR );
					if(t110.rows>frame.rows-10 || t110.cols>frame.cols-10)
					{
						t110=t100.clone();
					}

					Mat _templateEdge100 = EdgeEnhancemen(t100);
					Mat _templateEdge110 = EdgeEnhancemen(t110);
					Mat _templateEdge90 = EdgeEnhancemen(t90);


					//Mat _templateEdge100 = t100.clone();
					//Mat _templateEdge110 = t110.clone();
					//Mat _templateEdge90 = t90.clone();


					Mat _searchWindowEdge = EdgeEnhancemen(frame_gray(searchWin[k]));
					//Mat _searchWindowEdge = frame_gray;

					Mat result100;
					Mat result110;
					Mat result90;
					matchTemplate(_searchWindowEdge,_templateEdge100,result100,TM_CCORR_NORMED);
					matchTemplate(_searchWindowEdge,_templateEdge110,result110,TM_CCORR_NORMED);
					matchTemplate(_searchWindowEdge,_templateEdge90,result90,TM_CCORR_NORMED);

					normalize( result100, result100, 0, 1, NORM_MINMAX, -1, Mat() );
					normalize( result110, result110, 0, 1, NORM_MINMAX, -1, Mat() );
					normalize( result90, result90, 0, 1, NORM_MINMAX, -1, Mat() );

					/// Localizing the best match with minMaxLoc
					double minVal100; double maxVal100; Point minLoc100; Point maxLoc100;
					double minVal110; double maxVal110; Point minLoc110; Point maxLoc110;
					double minVal90; double maxVal90; Point minLoc90; Point maxLoc90;
					minMaxLoc( result100, &minVal100, &maxVal100, &minLoc100, &maxLoc100, Mat() );
					minMaxLoc( result110, &minVal110, &maxVal110, &minLoc110, &maxLoc110, Mat() );
					minMaxLoc( result90, &minVal90, &maxVal90, &minLoc90, &maxLoc90, Mat() );

					maxVal100=roundMax(maxVal100);
					maxVal110=roundMax(maxVal110);
					maxVal90=roundMax(maxVal90);

					Point maxLoc;
					Mat t;
					int Cmax= -1;
					if(maxVal100>Cmax)
					{
						maxLoc=maxLoc100;
						t=t100;
						Cmax=maxVal100;
					}
					if(maxVal90>Cmax)
					{
						maxLoc=maxLoc90;
						t=t90;
						Cmax=maxVal90;
					}
					if(maxVal110>Cmax)
					{
						maxLoc=maxLoc110;
						t=t110;
						Cmax=maxVal110;
					}
					Rect LocRect(Point(maxLoc.x+searchWin[k].x,maxLoc.y+searchWin[k].y),Point( maxLoc.x + t.cols + searchWin[k].x , maxLoc.y + t.rows + searchWin[k].y ));

					//--- update template
					if(Cmax>T3)
					{ 
						float landa = 0.5;
						Mat temp=frame_gray(LocRect);
						if(_template[k].rows == temp.rows && _template[k].cols == temp.cols)
						{
							addWeighted( _template[k], (1-landa), frame_gray(LocRect), landa, 0, _template[k] );
						}
						else // size not match
						{
							_template[k].release();
							_template[k] = temp.clone();

						}
						//addWeighted( _template, (1-landa)*Cmax, frame_gray(LocRect), landa*Cmax, 0, _template );

						templateLoc[k] = Rect(LocRect);
						//--- choose search window
						searchWin[k] = chooseSearchWindow( frame,  templateLoc[k]);

					}
					//else 
					////Don't update the template

					/// Show me what you got
					rectangle( frame, templateLoc[k], Scalar::all(0), 2, 8, 0 );

					//int minT = 24;//doubt?!
					//if(templateLoc[k].height<minT || templateLoc[k].width<minT)
					//{
					//	isMissed=true;
					//}

				}

			}
		}
		imshow( "", frame );
		ii++;
		int c = waitKey(10);
		if( (char)c == 27 ) { break; } // escape
	}

	waitKey(0);
	return 0;
}

//#include "opencv2/objdetect.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
//
//
//// this class is for OpenCV ParallelLoopBody
//class Parallel_pixel_SetToZero : public ParallelLoopBody
//{
//private:
//	uchar *p ;
//public:
//	Parallel_pixel_SetToZero(uchar* ptr ) : p(ptr) {}
//
//	virtual void operator()( const Range &r ) const
//	{
//		for ( register int i = r.start; i != r.end; ++i)
//		{
//			if(p[i]<50)
//				p[i]=0;
//		}
//	}
//};
//
//
//Mat EdgeEnhancemen(Mat src_gray)
//{
//	int scale = 1;
//	int delta = 0;
//	int ddepth = CV_16S;
//	Mat grad;
//	Mat nr_grad;
//	// Generate grad_x and grad_y
//	Mat grad_x, grad_y;
//	Mat abs_grad_x, abs_grad_y;
//
//	//--- standard horizontal and vertical Sobel 
//	// Gradient X
//	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//	convertScaleAbs( grad_x, abs_grad_x );
//	grad_x.release();
//
//	// Gradient Y
//	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//	convertScaleAbs( grad_y, abs_grad_y );
//	grad_y.release();
//	//-
//
//	// gradient magnitude image is obtained by summing two resultant images
//	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
//	abs_grad_x.release();
//	abs_grad_y.release();
//
//	// normalization is applied to stretch the pixel values in the whole range [0–255]
//	normalize(grad,nr_grad,0,255,NORM_MINMAX);
//	Mat new1= grad.clone();
//	Mat new2= nr_grad.clone();
//	return nr_grad;
//}
//
//
//double PowerOfRegion(Mat frame_gray)
//{
//	int scale = 1;
//	int delta = 0;
//	int ddepth = CV_16S;
//	Mat frame_gray_guss=frame_gray.clone();
//	/// reduce noise
//	//GaussianBlur( frame_gray, frame_gray_guss, Size(3,3), 0, 0, BORDER_DEFAULT );
//
//
//	/// Generate grad_x and grad_y
//	Mat grad_x, grad_y;
//	Mat abs_grad_x, abs_grad_y;
//
//	/// Gradient X
//	Sobel( frame_gray_guss, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//	convertScaleAbs( grad_x, abs_grad_x );
//
//	/// Gradient Y
//	Sobel( frame_gray_guss, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//	convertScaleAbs( grad_y, abs_grad_y );
//
//	int NR=frame_gray_guss.rows*frame_gray_guss.cols;
//
//	//--- set the gray-levels of less than 50 to zero 
//	uchar* p3 = abs_grad_x.data ;
//	parallel_for_( Range(0,NR) , Parallel_pixel_SetToZero(p3)) ;
//
//	uchar* p4 = abs_grad_y.data ;
//	parallel_for_( Range(0,NR) , Parallel_pixel_SetToZero(p4)) ;
//	//--
//
//	float PR;
//	Mat pow2_grad_x,pow2_grad_y;
//
//	pow(abs_grad_x,2.0,pow2_grad_x);
//	pow(abs_grad_y,2.0,pow2_grad_y);
//
//	PR=(1/NR)*sum(pow2_grad_x).val[0]+(1/NR)*sum(pow2_grad_y).val[0];
//
//	return PR;
//}
//
//
//Rect chooseSearchWindow(Mat frame, Rect templateLoc)
//{
//	int w=floor(templateLoc.width/2)+10,
//		h=floor(templateLoc.height/2)+10;
//
//	int X=templateLoc.x-w,	W=templateLoc.width+2*w,
//		Y=templateLoc.y-h,	H=templateLoc.height+2*h,
//		diffX=0,
//		diffY=0;
//	if(X<0)
//	{
//		diffX=abs(X);
//		X=0;
//	}
//	if(Y<0)
//	{
//		diffY=abs(Y);
//		Y=0;
//	}
//	W+=diffX;
//	H+=diffY;
//	if(X+W>frame.cols)
//		W=frame.cols-X;
//	if(Y+H>frame.rows)
//		H=frame.rows-Y;
//	return Rect(Point(X,Y),Point(X+W,Y+H));
//}
//float roundf(float x)
//{
//   return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
//}
//float roundMax( float val)
//{
//	return roundf(val * 1000) / 1000;
//}
//
//
///** Global variables */
//String cascade_name = "airplanes22stage.xml";
//CascadeClassifier cascade;
//
//int ii=0;
///** @function main */
//int main( void )
//{
//	VideoCapture capture;
//	Mat frame;
//
//	//-- Load the cascades
//	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
//
//	//-- Read the video stream
//	std::string video_name="Airplane7.avi";
//	capture.open( video_name);
//	if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
//
//
//	bool isMissed=true;
//	Rect templateLoc;
//	Mat _template;
//	Rect searchWin;
//	while (  capture.read(frame) )
//	{
//		if( frame.empty() )
//		{
//			printf(" --(!) No captured frame -- Break!");
//			break;
//		}
//
//		Mat frame_gray;
//		cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//
//		if(isMissed)
//		{
//			std::vector<Rect> airplanes;
//			//-- Detect airplanes
//			cascade.detectMultiScale( frame_gray, airplanes, 1.1, 2,0|CASCADE_SCALE_IMAGE, Size(1, 1) );
//			if(airplanes.size()!=0)
//			{
//				//--- choose template
//				templateLoc=airplanes[0];
//				_template=frame_gray(templateLoc);
//				//--- choose search window
//				searchWin = chooseSearchWindow( frame,  templateLoc);
//				//rectangle( frame, searchWin, Scalar( 0, 255, 0 ));
//
//				isMissed=false;
//			}
//		}
//		else
//		{
//
//			bool isCrowded=false;
//			int flag=0;
//			float T1=70;//intensity thershold
//			float T2=0.15;//relative power thershold
//			float T3=0.84;
//			int w=15;// floor(1/4*templateLoc.width);
//			int h=w;// floor(1/4*templateLoc.height);
//
//			//calculate intensity measure --------------------------------------------
//			bool is4RegionInFrame=false;
//			if(templateLoc.y-h>=1 
//				&& templateLoc.x-w>=1
//				&& templateLoc.y+templateLoc.height+w<frame_gray.rows
//				&& templateLoc.x+templateLoc.width+w<frame_gray.cols)
//			{
//				is4RegionInFrame=true;
//			}
//
//			if(is4RegionInFrame)
//			{
//				Rect up;
//				up = Rect(Point(templateLoc.x,templateLoc.y-h),Point(templateLoc.x+templateLoc.width,templateLoc.y));
//				rectangle( frame, up, Scalar( 0, 0, 255 ));
//
//				Rect left;
//				left=Rect(Point(templateLoc.x-w,templateLoc.y),Point(templateLoc.x,templateLoc.y+templateLoc.height));
//				rectangle( frame, left, Scalar( 0, 0, 255 ));		
//
//				Rect bottum(Point(templateLoc.x,templateLoc.y+templateLoc.height),Point(templateLoc.x+templateLoc.width,templateLoc.y+templateLoc.height+w));
//				rectangle( frame, bottum, Scalar( 0, 0, 255 ));		
//
//				Rect right(Point(templateLoc.x+templateLoc.width,templateLoc.y),Point(templateLoc.x+templateLoc.width+w,templateLoc.y+templateLoc.height));
//				rectangle( frame, right, Scalar( 0, 0, 255 ));
//
//				//-- display search window
//				//rectangle( frame, searchWin, Scalar( 0, 255, 0 ));
//
//				Mat _up= frame_gray( up );
//				Mat _left= frame_gray( left );
//				Mat _bottum= frame_gray( bottum );
//				Mat _right= frame_gray( right );
//
//				Mat _UB;
//				absdiff(_up,_bottum,_UB);
//				float UB=mean(_UB).val[0];
//
//				Mat _LR;
//				absdiff(_left,_right,_LR);
//				float LR= mean(_LR).val[0];
//				//end - calculate intensity measure --------------------------------------------
//
//
//				if(UB>T1 || LR>T1) // intensity
//				{	
//					cout<<UB<<"\t";
//					cout<<LR<<"\t"<<"crowded"<<endl;
//					isCrowded=true;
//					//cout<<isCrowded<<"\t"<<ii<<endl;
//				}
//				else // calculate relative power
//				{
//					float PR_u=PowerOfRegion(_up);
//					float PR_l=PowerOfRegion(_left);
//					float PR_b=PowerOfRegion(_bottum);
//					float PR_r=PowerOfRegion(_right);
//					float PR_t=PowerOfRegion(_template);
//					cout<<PR_u<<"\t"<<PR_l<<"\t"<<PR_b<<"\t"<<PR_r<<"\t"<<PR_t<<endl;
//
//					if((PR_u/PR_t)>T2 ||(PR_l/PR_t)>T2 ||(PR_b/PR_t)>T2 ||(PR_r/PR_t)>T2)
//					{
//						cout<<"relative power detected"<<"\t"<<"crowded"<<endl;
//						isCrowded=true;
//					}
//				}
//				//}
//
//			}
//			else
//			{
//				cout<<"rect out of frames"<<endl;
//			}
//
//
//			//if(isCrowded) // RMI method
//			//{
//			//	
//			//}
//			//else // EC method
//			{
//				if(ii==85)
//					cout<<endl;
//				Mat t90,t100,t110;
//				t100=_template.clone();
//				resize(t100, t110, Size(), 1.1, 1.1, CV_INTER_AREA );
//				resize(t100, t90, Size(), 0.9, 0.9, CV_INTER_LINEAR );
//				if(t110.rows>frame.rows-10 || t110.cols>frame.cols-10)
//				{
//					t110=t100.clone();
//				}
//
//				Mat _templateEdge100 = EdgeEnhancemen(t100);
//				Mat _templateEdge110 = EdgeEnhancemen(t110);
//				Mat _templateEdge90 = EdgeEnhancemen(t90);
//
//
//				//Mat _templateEdge100 = t100.clone();
//				//Mat _templateEdge110 = t110.clone();
//				//Mat _templateEdge90 = t90.clone();
//
//
//				Mat _searchWindowEdge = EdgeEnhancemen(frame_gray(searchWin));
//				//Mat _searchWindowEdge = frame_gray;
//
//				Mat result100;
//				Mat result110;
//				Mat result90;
//				matchTemplate(_searchWindowEdge,_templateEdge100,result100,TM_CCORR_NORMED);
//				matchTemplate(_searchWindowEdge,_templateEdge110,result110,TM_CCORR_NORMED);
//				matchTemplate(_searchWindowEdge,_templateEdge90,result90,TM_CCORR_NORMED);
//
//				normalize( result100, result100, 0, 1, NORM_MINMAX, -1, Mat() );
//				normalize( result110, result110, 0, 1, NORM_MINMAX, -1, Mat() );
//				normalize( result90, result90, 0, 1, NORM_MINMAX, -1, Mat() );
//
//				/// Localizing the best match with minMaxLoc
//				double minVal100; double maxVal100; Point minLoc100; Point maxLoc100;
//				double minVal110; double maxVal110; Point minLoc110; Point maxLoc110;
//				double minVal90; double maxVal90; Point minLoc90; Point maxLoc90;
//				minMaxLoc( result100, &minVal100, &maxVal100, &minLoc100, &maxLoc100, Mat() );
//				minMaxLoc( result110, &minVal110, &maxVal110, &minLoc110, &maxLoc110, Mat() );
//				minMaxLoc( result90, &minVal90, &maxVal90, &minLoc90, &maxLoc90, Mat() );
//				
//				maxVal100=roundMax(maxVal100);
//				maxVal110=roundMax(maxVal110);
//				maxVal90=roundMax(maxVal90);
//
//				Point maxLoc;
//				Mat t;
//				int Cmax= -1;
//				if(maxVal100>Cmax)
//				{
//					maxLoc=maxLoc100;
//					t=t100;
//					Cmax=maxVal100;
//				}
//				if(maxVal90>Cmax)
//				{
//					maxLoc=maxLoc90;
//					t=t90;
//					Cmax=maxVal90;
//				}
//				if(maxVal110>Cmax)
//				{
//					maxLoc=maxLoc110;
//					t=t110;
//					Cmax=maxVal110;
//				}
//				Rect LocRect(Point(maxLoc.x+searchWin.x,maxLoc.y+searchWin.y),Point( maxLoc.x + t.cols + searchWin.x , maxLoc.y + t.rows + searchWin.y ));
//
//				//--- update template
//				if(Cmax>T3)
//				{ 
//					float landa = 0.5;
//					Mat temp=frame_gray(LocRect);
//					if(_template.rows == temp.rows && _template.cols == temp.cols)
//					{
//						addWeighted( _template, (1-landa), frame_gray(LocRect), landa, 0, _template );
//					}
//					else // size not match
//					{
//						_template.release();
//						_template = temp.clone();
//
//					}
//					//addWeighted( _template, (1-landa)*Cmax, frame_gray(LocRect), landa*Cmax, 0, _template );
//
//					templateLoc = Rect(LocRect);
//					//--- choose search window
//					searchWin = chooseSearchWindow( frame,  templateLoc);
//
//				}
//				//else 
//				////Don't update the template
//
//				/// Show me what you got
//				rectangle( frame, templateLoc, Scalar::all(0), 2, 8, 0 );
//
//				int minT = 24;//doubt?!
//				if(templateLoc.height<minT || templateLoc.width<minT)
//				{
//					isMissed=true;
//				}
//
//			}
//
//		}
//		imshow( "", frame );
//		ii++;
//		int c = waitKey(10);
//		if( (char)c == 27 ) { break; } // escape
//	}
//
//	waitKey(0);
//	return 0;
//}

