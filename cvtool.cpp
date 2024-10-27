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

#ifdef _WIN32
#ifndef _WINUSER_
extern "C" int __stdcall MessageBoxA(void*, const char*, const char*, int);
#endif
#endif

int main(int argc, char** argv)
{
    cout << "Usage: cvtool <img> <filter,...>\n";
	if (argc != 3)
	{
		if (argc == 1)
		{
#ifdef _WIN32
			::MessageBoxA(NULL, "Usage: cvtool <img> <filter,...>\n", "cvtool usgaes", 1);
#else
			std::string message = "Usage: cvtool <img> <filter,...>\n";
			Mat img = cv::Mat::zeros(cv::Size(800, 60), CV_8UC3);
			// Define font and calculate text size
			int font = cv::FONT_HERSHEY_SIMPLEX;
			int fontScale = 1;
			int thickness = 2;
			cv::Size textSize = cv::getTextSize(message, font, fontScale, thickness, 0);

			// Calculate text position to center it on the image
			int textX = (img.cols - textSize.width) / 2;
			int textY = (img.rows + textSize.height) / 2;

			// Put the text onto the image
			cv::putText(img, message, cv::Point(textX, textY), font, fontScale, cv::Scalar(255, 255, 255), thickness);

			// Display the image as a message box
			cv::imshow("cvtool usgaes", img);
			cv::waitKey(0);
			cv::destroyAllWindows();
#endif
		}
		return -1;
	}

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
            Mat res;
            res = fg.filter(frame);
            do
			{
            	c = waitKey(0);
				switch (c)
				{
				case '1':case '2':case '3':case '4':case '5':case '6':
				case '7':case '8':case '9':
					fg.bringTop(c - '1');
					break;
				case '=':
					fg.bringTop();
					break;
				}
        	} while (c != 'q'
               		&& c != 'Q'
               		&& c != ' ');
		}
    }
    return 0;
}
