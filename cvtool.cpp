/**
MIT License
Copyright (c) 2020 bbqz007 <https://github.com/bbqz007, http://www.cnblogs.com/bbqzsl>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "cvtool.h"
using namespace zhelper;
using namespace cvtool;

int main(int argc, char** argv)
{
    cout << "Usage: cvtool <img> <filter,...>\n";
    if (argc != 3)
        return -1;

    filter_graph fg;
    fg.open(argv[2]);
    VideoCapture cap;
    cap.open(argv[1]);
    Mat frame;
    char c = '\0';
    while((cap >> frame, !frame.empty())
          && (c != 'q' && c != 'Q'))
    {
        {
#if 1
            Mat res;
            res = fg.filter(frame);
#else
            fg.apply([&](Mat res){
                Mat pyr, timg, gray0(frame.size(), CV_8U), gray;
                int thresh = 50, N = 11;
                vector<vector<Point> > contours;
                vector<vector<Point> > squares;
                // find squares in every color plane of the image
                for( int c = 0; c < 3; c++ )
                {
                    int ch[] = {c, 0};
                    mixChannels(&res, 1, &gray0, 1, ch, 1);

                    ostringstream os;
                    imshow((os << "channel-" << (int)c, os).str(), gray0);

                    // try several threshold levels
                    for( int l = 0; l < N; l++ )
                    {
                        // hack: use Canny instead of zero threshold level.
                        // Canny helps to catch squares with gradient shading
                        if( l == 0 )
                        {
                            // apply Canny. Take the upper threshold from slider
                            // and set the lower to 0 (which forces edges merging)
                            Canny(gray0, gray, 0, thresh, 5);
                            // dilate canny output to remove potential
                            // holes between edge segments
                            dilate(gray, gray, Mat(), Point(-1,-1));
                        }
                        else
                        {
                            // apply threshold if l!=0:
                            //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                            gray = gray0 >= (l+1)*255/N;
                        }

                        // find contours and store them all as a list
                        findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

                        vector<Point> approx;

                        // test each contour
                        for( size_t i = 0; i < contours.size(); i++ )
                        {
                            // approximate contour with accuracy proportional
                            // to the contour perimeter
                            approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

                            // square contours should have 4 vertices after approximation
                            // relatively large area (to filter out noisy contours)
                            // and be convex.
                            // Note: absolute value of an area is used because
                            // area may be positive or negative - in accordance with the
                            // contour orientation
                            if( approx.size() == 4 &&
                                fabs(contourArea(approx)) > 1000 &&
                                isContourConvex(approx) )
                            {
                                double maxCosine = 0;

                                for( int j = 2; j < 5; j++ )
                                {
                                    struct lambda
                                    {
                                        static double angle( Point pt1, Point pt2, Point pt0 )
                                        {
                                            double dx1 = pt1.x - pt0.x;
                                            double dy1 = pt1.y - pt0.y;
                                            double dx2 = pt2.x - pt0.x;
                                            double dy2 = pt2.y - pt0.y;
                                            return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
                                        }
                                    };

                                    // find the maximum cosine of the angle between joint edges
                                    double cosine = fabs(lambda::angle(approx[j%4], approx[j-2], approx[j-1]));
                                    maxCosine = MAX(maxCosine, cosine);
                                }

                                // if cosines of all angles are small
                                // (all angles are ~90 degree) then write quandrange
                                // vertices to resultant sequence
                                if( maxCosine < 0.3 )
                                    squares.push_back(approx);
                            }
                        }
                    }
                    //drawContours(frame.clone(), contours, static_cast<int>(contours.size()) - 1, Scalar(0,255,0), FILLED);
                }

                Mat show = frame.clone();
                polylines(show, squares, true, Scalar(0, 255, 0), 3, LINE_AA);
                imshow("apply", show);
            }).filter(frame);
#endif
			do
			{
            	c = waitKey(0);
        	} while (c != 'q'
               		&& c != 'Q'
               		&& c != ' ');
		}
    }
    return 0;
}
