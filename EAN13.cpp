#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map> 
#include <string> 
using namespace std;
using namespace cv;


/* Left Character*/
typedef   struct   itemstruct 
  { 
            char    a[20]; 
            char   b[20]; 
  }itemS;
  itemS  s[20]   =   {{"0","A"},{"0","B"},{"1","A"},{"1","B"},{"2","A"},{"2","B"},{"3","A"},{"3","B"},{"4","A"},{"4","B"},{"5","A"},{"5","B"},{"6","A"},{"6","B"},{"7","A"},{"7","B"},
      {"8","A"},{"8","B"},{"9","A"},{"9","B"} 
                      }; 
  

 
class ImageProcess
{
private:
    double Avg[180]; 

public:
    ImageProcess();
    cv::Mat getRotateImage(const cv::Mat &image,int angle);
    void countAvg(const cv::Mat &image); //count the average of bars’ height
    int getAngle();
    cv::Mat getFinalImage(const cv::Mat &image, int angle);
  
};


ImageProcess::ImageProcess()
{
 ;
}


cv::Mat ImageProcess::getRotateImage(const cv::Mat &image,int angle)
{
    cv::Mat matTmp = image.clone();
    cv::Point2f center = cv::Point2f(matTmp.cols / 2, matTmp.rows / 2);  
    double scale = 1;  
    cv::Mat rotateMat;   
    rotateMat = cv::getRotationMatrix2D(center, angle, scale);  
    cv::Mat rotateImg;
    cv::warpAffine(matTmp, rotateImg, rotateMat, matTmp.size());  
    Mat rio_Image=rotateImg(Range(matTmp.rows  / 2-50,matTmp.rows  / 2+50),Range(matTmp.cols/ 2-50,matTmp.cols/ 2+50));
    return rio_Image;
}

//count bars' average
void ImageProcess::countAvg(const cv::Mat &image)
{
    cv::Mat matTmp = image.clone();
    for(int i=0; i<180; i++)
    {
       int num=0;
       int tmp = 0;
       int a = 0;
       cv::Mat RotateImage=getRotateImage(matTmp, i);
       int height = RotateImage.rows, width = RotateImage.cols;
    
      for (int col = 0; col < width; col++)
      {
          int tmp = 0;
        
          for (int row = 0; row < height; row++)
          {
              if (RotateImage.at<uchar>(row, col) == 0)
              {
                  tmp++;
              }

          }
          if (tmp!= 0 )
          {
              num++;
          }
       
          a=a+tmp;
      }

      Avg[i]=a*1.0/num;
      }
  }


int ImageProcess:: getAngle() 
{   
    int max=0;
    for(int i=0; i<180; i++)
    {
    
        if (Avg[i+1]>Avg[max])
        {
            max=i+1;
        }
    }
    return max;
}
//rotation according to certain angle
cv::Mat ImageProcess::getFinalImage(const cv::Mat &image,int angle)
{
    cv::Mat matTmp = image.clone();
    cv::Point2f center = cv::Point2f(matTmp.cols / 2, matTmp.rows / 2);   
    double scale = 1;   
    cv::Mat rotateMat;   
    rotateMat = cv::getRotationMatrix2D(center, angle,scale);
    cv::Mat rotateImg;
    cv::warpAffine(matTmp, rotateImg, rotateMat, matTmp.size());
    Mat rotateImg_binary;
    cv::threshold(rotateImg, rotateImg_binary, 128, 255, CV_THRESH_BINARY);  
    return rotateImg_binary;
}



class BarsDecode
{
    

public:
    BarsDecode();
    cv::Mat getVerProjImage(const cv::Mat &image);
    void getCode(const cv::Mat &image, Mat &OriginalImage);
    itemS findLeft (string code1);
    char findRight (string code2);
    char findFirst (string code3);
};

BarsDecode::BarsDecode()
{
}


//get barcode area using principle projection

cv::Mat BarsDecode::getVerProjImage(const cv::Mat &image)
{
    cv::Mat matTmp = image.clone();
    int height = matTmp.rows, width = matTmp.cols;//图像的高和宽
    int tmp = 0;
    int *projArray = new int[width];

    for (int col = 0; col < width; col++)
    {
        tmp = 0;
        
        for (int row = 0; row < height; row++)
        {
            if (matTmp.at<uchar>(row, col) == 0)// count black pixel
            {
                tmp++;
            }       
        }
        projArray[col] = tmp;
  
    }

    int star=0;
    
    for (int k=0; k<width-6; k++)
    {
        if ((projArray[k]==0)&(projArray[k+1]==0)&(projArray[k+2]==0)&(projArray[k+3]==0)&(projArray[k+4]==0)&(projArray[k+5]==0)&(projArray[k+6]!=0))
            {
                star=k+6;
                cout<<"the beginning of bars starts at "<<star<<"col"<<endl;
        break;}
    }
    int end=0;
    for (int k=width-7;k > 0;k--)
    {
        if ((projArray[k]!=0)&(projArray[k+1]==0)&(projArray[k+2]==0)&(projArray[k+3]==0)&(projArray[k+4]==0)&(projArray[k+5]==0)&(projArray[k+6]==0))
            {
                end=k;
                cout<<"the beginning of bars ends at "<<end<<"col"<<endl<<endl;
        break;}
    }

    int widthBarsArea=end-star;
    cv::Mat projImg(height, widthBarsArea,  CV_8U, cv::Scalar(255));//255白底

    for (int col = star; col < end+1; ++col)
    {
        cv::line(projImg, cv::Point(col-star, height - projArray[col]), cv::Point(col-star, height), cv::Scalar::all(0));
    
    }
    imshow("[BarsAreaVerticalProjection]",projImg);

    Mat BarsImage= matTmp(Range::all(),Range(star-10,end+10));

    delete[] projArray;

    
   return  BarsImage;
}

//convert to 0-1 sets
void BarsDecode::getCode(const cv::Mat &image, Mat &OriginalImage)
{

     cv::Mat matTmp = image.clone();
     int Imgheight = matTmp.rows, Imgwidth = matTmp.cols;
     int num=0;
     vector <vector <int>> width(Imgheight); 
            for(int   i=0;i <Imgheight;i++) 
                width[i].resize(60);
     vector <vector <int>> total_len(Imgheight); 
            for(int   i=0;i <Imgheight;i++) 
                total_len[i].resize(60);
     int color_binary[59];
    

            for(int i=0;i<Imgheight;i++)
            {
                int *pos=new int [Imgwidth];
                int pos_cnt=0,width_id=0;

                for(int j=0;j<Imgwidth-1;j++)
                {
                    int a=matTmp.at<uchar>(i, j);
                    int b=matTmp.at<uchar>(i,j+1);

                    if ( a!=b )
                    {
                        
                        pos[pos_cnt]=j;

                        if (pos_cnt>0)
                        {
                          width[i][width_id]=pos[pos_cnt]-pos[pos_cnt-1];
                          
                          if (matTmp.at<uchar>(i, j)==0)
                                   color_binary[width_id] = 1;
                          else
                                   color_binary[width_id] = 0;
                          
                          width_id++;
                         }
                                    
                             pos_cnt=pos_cnt+1;
                    }
                    
                }
            
                if (width_id==59)
                {
                  
                   for (int k=0;k<59;k++)
                   {
                      

                       total_len[num,k]=width[i,k];
                   }
                   
                   ++num;

                }

                delete pos;
            }

        
                double final_width[59];
                for (int m=0;m<59;m++)
                {     
                        int sum=0;
                        for(int n=0;n<num;n++)
                                sum=sum+total_len[n][m];
                        final_width[m]=sum/59.0;
                }


            double sum_width=0;
            for (int p=0;p<59;p++)
            {
                
                sum_width=sum_width+final_width[p];
            }


            double Avg_model;
            Avg_model=sum_width/95.0;
            

            int model_num[59];
            for (int w=0; w<59; w++)
            {
                model_num[w] =(int) (final_width[w]/Avg_model+0.5);
                cout<<"model_num"<<w<<":"<<model_num[w]<<"   ";
            }
            cout<<endl;
            
            
            
            int code[95]; 
            int mm=0;
            for (int q=0;q<59;q++)
            {
            
                if ( model_num[q] == 1 )
                    {
                        code[mm] = color_binary[q];
                        mm++;
                        continue;
                }
                if ( model_num[q] == 2 )
                    {
                        code[mm] = color_binary[q];
                        code[mm+1] = color_binary[q];
                        mm=mm+2;
                        continue;
                }
                    
                if ( model_num[q] == 3 )
                        {
                        code[mm] = color_binary[q];
                        code[mm+1] = color_binary[q];
                        code[mm+2] = color_binary[q];
                        mm=mm+3;
                        continue;
                }

                if ( model_num[q] == 4 )
                    {
                        code[mm] = color_binary[q];
                        code[mm+1] = color_binary[q];
                        code[mm+2] = color_binary[q];
                        code[mm+3] = color_binary[q];
                        mm=mm+4;
                        continue;
                }
            

            }
            for (int e=0; e<95; e++)
                cout<<code[e]<<"  ";

          //check
          int check_num=0;
          if (code[0]==1 && code[1]==0 && code[2]==1 && code[45]==0 && code[46]==1 && code[47]==0 && code[48]==1 && code[49]==0 && code[92]==1 && code[93]==0 && code[94]==1 )
             check_num=1;
          else
              check_num=0;
          if ( check_num==0 )
              cout<<"fail!please finde the code again!"<<endl;




         
         char odd_even[7];
         char final_number[14] = {'\0'};
    
         
          for(int i=3,m=0;i<45;i=i+7,m++)
              {
                  
                  char cha[8];
                  sprintf(cha,"%d",code[i]);
                  sprintf(cha+1,"%d",code[i+1]);
                  sprintf(cha+2,"%d",code[i+2]);
                  sprintf(cha+3,"%d",code[i+3]);
                  sprintf(cha+4,"%d",code[i+4]);
                  sprintf(cha+5,"%d",code[i+5]);
                  sprintf(cha+6,"%d",code[i+6]);
                  itemS Final_left;
                  Final_left = findLeft(cha);
                
                 odd_even[m]= *(Final_left.b);
                 
                  final_number[m+1]=*(Final_left.a);
                  

            }
          
          for(int i=50,m=6;i<92;i=i+7,m++)
          {
                char cha2[8];
                sprintf(cha2,"%d",code[i]);
                sprintf(cha2+1,"%d",code[i+1]);
                sprintf(cha2+2,"%d",code[i+2]);
                sprintf(cha2+3,"%d",code[i+3]);
                sprintf(cha2+4,"%d",code[i+4]);
                sprintf(cha2+5,"%d",code[i+5]);
                sprintf(cha2+6,"%d",code[i+6]);
                char Final_right; 
                Final_right=findRight(cha2);
                final_number[m+1]=Final_right;
                
          }
          
          
          odd_even[6]='\0';
          final_number[0] = findFirst(odd_even);        
          
          int oddsum=0,evensum=0,sum=0,first_number_int=0,ture_checkcode=0;
          int final_number_int[14];
          for ( int i= 0;i<14;i++)
          {
              final_number_int[i] = final_number[i] - '0';
          }
          
          for (int i=1;i<13;i=i+2)
          {

              oddsum=oddsum+final_number_int[i];//even(B)

          }
          for (int i=0;i<11;i=i+2)
          {
              evensum=evensum+final_number_int[i];//odd(A)
          }
        
          sum=oddsum*3+evensum;
          ture_checkcode=10-(sum%10);
          cout<<endl<<"ture_checkcode:"<<ture_checkcode<<endl;
          if (ture_checkcode == final_number_int[12])       
              {cout<<"THE INFORMATION IS TRUE!"<<endl<<"ALREADY GET BARCODE INFORMATION:"<<endl;
            
              for (int i=0;i<14;i++)
              {
                  cout<<final_number[i];
              }           
              putText(OriginalImage, final_number, Point(30, 20), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0, 255));
          }
          else
              {
                  
                  int code2[95];
                  char odd_even_agn[7]= {'\0'};;
                  char final_number_agn[14]= {'\0'};

                  for ( int i=0;i<95;i++)
                  {
                      code2[i] = code[94-i];
                  }
                  for ( int i=0;i<95;i++)
                  {
                      cout<<code2[i];
                  }cout<<endl;

                   
          for(int i=3,m=0;i<45;i=i+7,m++)
              {
                  
                  char cha_agn[8];
                  sprintf(cha_agn,"%d",code2[i]);
                  sprintf(cha_agn+1,"%d",code2[i+1]);
                  sprintf(cha_agn+2,"%d",code2[i+2]);
                  sprintf(cha_agn+3,"%d",code2[i+3]);
                  sprintf(cha_agn+4,"%d",code2[i+4]);
                  sprintf(cha_agn+5,"%d",code2[i+5]);
                  sprintf(cha_agn+6,"%d",code2[i+6]);
                
                 
                  itemS Final_left_agn;
                  Final_left_agn = findLeft(cha_agn);
                
                 odd_even_agn[m]= *(Final_left_agn.b);
                 
                  final_number_agn[m+1]=*(Final_left_agn.a);
                  

            }
          
          for(int i=50,m=6;i<92;i=i+7,m++)
          {
                char cha2_agn[8];
                sprintf(cha2_agn,"%d",code2[i]);
                sprintf(cha2_agn+1,"%d",code2[i+1]);
                sprintf(cha2_agn+2,"%d",code2[i+2]);
                sprintf(cha2_agn+3,"%d",code2[i+3]);
                sprintf(cha2_agn+4,"%d",code2[i+4]);
                sprintf(cha2_agn+5,"%d",code2[i+5]);
                sprintf(cha2_agn+6,"%d",code2[i+6]);
                char Final_right_agn; 
                Final_right_agn=findRight(cha2_agn);
                final_number_agn[m+1]=Final_right_agn;
                
          }
          
          
          final_number_agn[0] = findFirst(odd_even_agn);     
          //check-sum digit
          int oddsum_agn=0,evensum_agn=0,sum_agn=0,first_number_int_agn=0,ture_checkcode_agn=0;
          int final_number_int_agn[14];
          for ( int i= 0;i<14;i++)
          {
              final_number_int_agn[i] = final_number_agn[i] - '0';
          }
          
          for (int i=1;i<13;i=i+2)
          {

              oddsum_agn=oddsum_agn+final_number_int_agn[i];//even(B)

          }
          for (int i=0;i<11;i=i+2)
          {
              evensum_agn=evensum_agn+final_number_int_agn[i];//odd(A)
          }
        
          sum_agn=oddsum_agn*3+evensum_agn;
          ture_checkcode_agn=10-(sum_agn%10);
         if (ture_checkcode_agn == final_number_int_agn[12])        //校验位 最后一位
              {cout<<"THE BARCODE IS OPPOSITE!(THE REAL ANGLE IS 180))"<<endl<<"ALREADY GET BARCODE INFORMATION:"<<endl;
            
              for (int i=0;i<13;i++)
              {
                  cout<<final_number_agn[i];
              }           
              putText(OriginalImage, final_number_agn, Point(30, 20), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255, 255)); 
              string CHN_s = "(if you see this sentence,bars' angle is 180)";
              putText(OriginalImage, CHN_s, Point(30, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0, 255));
          }
          else
                  cout<<"ERROR! CANNOT GET THE TRUE INFORMATION"<<endl;


          }

                 
          
}
    
//check left-hand code encoding table
itemS BarsDecode::findLeft(string codestr)
{
           
            map<string, itemS> mymap; 
            string  str[20]   =   {"0001101","0100111","0011001","0110011","0010011","0011011","0111101","0100001","0100011","0011101","0110001","0111001","0101111","0000101","0111011","0010001","0110111","0001001","0001011","0010111"}; 
            
             for(int i = 0;   i<20;   i++) 
            { 
                 mymap.insert(make_pair(str[i], s[i])); 
            } 
            itemS Left_pair;
            map<string, itemS>::iterator l_it;; 
            l_it=mymap.find(codestr);
            if(l_it==mymap.end())
                cout<<"we do not find "<<codestr<<endl;
            else 
             Left_pair=l_it->second;
            return Left_pair;

}

//check right-hand code encoding table
char BarsDecode::findRight(string codestr)
{
    
            char char_number[]={'0','1','2','3','4','5','6','7','8','9'};//char型用单引号
            map<string, char> mymap2;
            string  str[10]   =   {"1110010","1100110","1101100","1000010","1011100","1001110","1010000","1000100","1001000","1110100"}; 
            
             for(int i = 0;   i<10;   i++) 
            { 
                 mymap2.insert(make_pair(str[i],char_number[i])); 
            } 
            char right_number;
            map<string, char>::iterator l_it;; 
            l_it=mymap2.find(codestr);
            if(l_it==mymap2.end())
                cout<<"we do not find "<<codestr<<endl;
            else 
             right_number=l_it->second;
            return right_number;
}

//check first code encoding table
char BarsDecode::findFirst (string codestr)
{
    char char_number[]={'0','1','2','3','4','5','6','7','8','9'};
    map<string, char> mymap3;
    string  str[10]   =   {"AAAAAA","AABABB","AABBAB","AABBBA","ABAABB","ABBAAB","ABBBAA","ABABAB","ABABBA","ABBABA"}; 
            
    for(int i = 0;   i<10;   i++) 
    { 
        mymap3.insert(make_pair(str[i],char_number[i])); 
        } 
    char first_number;
    map<string, char>::iterator l_it;; 
    l_it=mymap3.find(codestr);
    if(l_it==mymap3.end())
        cout<<"we do not find "<<codestr<<endl;
    else 
        first_number=l_it->second;
    return first_number;
}


void main()
{
    VideoCapture video("1.mp4");
    VideoWriter writer("OriginalImage.avi", -1, 1, Size(568, 320));

    if (!video.isOpened())              
        cout << "fail to open!" << endl;
    Mat OriginalImage; 
    int nframes = 0;
    char name[50];
    bool wFlag = true;
    int i = 0;
    Mat Gray_Img;


    while (1)
    {

        video >> OriginalImage; 

        if (nframes % 50 == 0) //extract a image per 20 frames
        {

            if (OriginalImage.empty())
            {
                cout << "Fail to load image" << endl;
                break;
            }
                    Mat OriginalImage =imread( "111.jpg");
            Mat Gray_Img;
            imshow("OriginalImage", OriginalImage);
            imwrite("OriginalImage.jpg",OriginalImage);
            
            if (OriginalImage.channels() > 1) 
            {
                cvtColor(OriginalImage, Gray_Img, COLOR_RGB2GRAY);
            }
            imwrite("Gray_Img.jpg",Gray_Img);
            Mat binaryImage;
            cv::threshold(Gray_Img, binaryImage, 0, 255, CV_THRESH_OTSU);//CV_THRESH_OTSU method

            cv::imshow("BinaryImage", binaryImage);
            imwrite("binaryImage.jpg",binaryImage);
            ImageProcess Find_angle;
            Find_angle.countAvg(binaryImage);
            int Angle = Find_angle.getAngle(); //get the barcode’s angle
            cout << "These bars' angle is" << Angle << endl << endl;
            if (Angle != 0)
            {
            char cAngle[4];
            itoa(Angle, cAngle, 10);
            putText(OriginalImage, cAngle, Point(180, 40), CV_FONT_HERSHEY_COMPLEX, 0.6, Scalar(0, 255, 255));
            string CHN = "bars' angle : ";
            putText(OriginalImage, CHN, Point(20, 40), CV_FONT_HERSHEY_COMPLEX, 0.6, Scalar(0, 255, 255));
            }
            else 
            {
            string CHN_special = "bars'angle:0 or 180 ";
            putText(OriginalImage, CHN_special, Point(50, 40), CV_FONT_HERSHEY_COMPLEX, 0.6, Scalar(0, 0, 255));
            }
            cv::Mat RotatedImage = Find_angle.getFinalImage(binaryImage, Angle);

            imshow("[RotatedImage]", RotatedImage);
            imwrite("rotated.jpg",RotatedImage);

            Mat cent_Image = RotatedImage(Range(RotatedImage.rows / 2 - 50, RotatedImage.rows / 2 + 50), Range::all());
            imwrite("cut1.jpg",cent_Image);
            BarsDecode bars_decode;
            cv::Mat BarsImage = bars_decode.getVerProjImage(cent_Image);

            bars_decode.getCode(BarsImage, OriginalImage);

            imshow("[BarsImage]", BarsImage);
            imshow("OriginalImage", OriginalImage);

            if (wFlag == true)
            {
                sprintf(name, "picture\\%d.jpg", i);
                imwrite(name, OriginalImage);
                writer << OriginalImage;
                waitKey(1);
            }

            waitKey(1);
            i++;
        }

        ++nframes;
    }
//waitKey(0);
}