#include "Util.h"


double getMemoryUsage(){
    struct rusage memInfo;

    getrusage(RUSAGE_SELF, &memInfo);

    double memoryInMB = (double)memInfo.ru_maxrss / 1024.0;
    return memoryInMB;
}
