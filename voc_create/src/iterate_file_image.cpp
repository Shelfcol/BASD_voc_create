#include<io.h>
#include<iostream>
#include <vector>
#include<string>
 
using namespace std;
 
void get_image_names(std::string file_path, std::vector<std::string>& file_names)
{
	intptr_t hFile = 0;
	_finddata_t fileInfo;
	hFile = _findfirst(file_path.c_str(), &fileInfo);
	if (hFile != -1){
		do{
                        //如果为文件夹，可以通过递归继续遍历，此处我们不需要
			if ((fileInfo.attrib &  _A_SUBDIR)){
				continue;
			}
                        //如果为单个文件直接push_back
			else{
				file_names.push_back(fileInfo.name);
				cout << fileInfo.name << endl;
			}
 
		} while (_findnext(hFile, &fileInfo) ==0);
 
		_findclose(hFile);
	}
}
 
 
int main()
{
	std::vector<std::string> file_names;
	get_image_names("/home/gxf/slam/slam_dataset/kitti/dataset3/sequences/00/000000.png", file_names);
 
	return 0;
}