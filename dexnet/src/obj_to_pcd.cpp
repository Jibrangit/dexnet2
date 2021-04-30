
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
// #include <pcl / io / vtk_lib_io.h> // loadPolygonFileOBJ header belongs;
 
using namespace pcl;
int main()
{
pcl::PolygonMesh mesh;
pcl::io::loadOBJFile("/home/jibran/catkin_ws/src/dexnet/obj/guitar.obj", mesh);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
pcl::io::savePCDFileASCII("/home/jibran/catkin_ws/src/dexnet/pcd/guitar.pcd", *cloud);
return 0;
}