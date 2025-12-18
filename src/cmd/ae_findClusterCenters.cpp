#include "cmd.h"
#include "dMRI/tractography/autoencoder/StreamlineAutoencoder.h"
#include "dMRI/tractography/autoencoder/autoEncoderUtils.h"
#include "nanoflann/nanoflann.hpp"

using namespace NIBR;

namespace CMDARGS_FINDCLUSTERCENTERS {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    float  maxDist;
    int    maxIteration     = 0;

    int    batchSize        = 1000000;
    int    miniBatchSize    = 1000;
    bool   randomize        = false;
    bool   shuffle          = false;
    bool   useCPU           = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    size_t newGenSize       = 0;
}

using namespace CMDARGS_FINDCLUSTERCENTERS;

void run_findClusterCenters()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(out_path) != "clc") {
        disp(MSG_ERROR,"Output cluster center file must have .clc extension.");
        return;
    }

    bool isFile = false;
    
    std::string ext = getFileExtension(inp_path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "bin") isFile = true;
    

    bool isFolder = existsFolder(inp_path);

    if (!isFile && !isFolder) {
        disp(MSG_ERROR,"Unknown input");
        return;
    }

    // Check input limits
    if (batchSize < 1) {
        disp(MSG_ERROR,"Minimum batchSize is 1");
        return;
    }

    if (miniBatchSize < 1) {
        disp(MSG_ERROR,"Minimum miniBatchSize is 1");
        return;
    }

    if (miniBatchSize > batchSize) {
        disp(MSG_ERROR,"Max miniBatchSize is batchSize");
        return;
    }
    

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(model_spec, useCPU, newGenSize);
    if (!model.isReady()) return;

    int latDimMultiplier = 2;
    if(model.newGenSize > 0) {
        latDimMultiplier = 1;
    }

    // Open binary files to read encoded streamlines
    std::vector<std::ifstream*> file;
    std::vector<int> scnt;
    std::vector<int> scnt_cumSum;
    int totalCnt = 0;
     
    // Add files to the list to process
    // keeps the count of streamlines
    auto addToProcess = [&](std::string fname) -> bool {
        std::ifstream* ifs = new std::ifstream(fname, std::ios::binary | std::ios::ate | std::ios::in);
        if (!ifs->is_open()) {
            disp(MSG_ERROR,"Failed to open file: %s", fname.c_str());
            return false;
        }
        std::streampos fileSize = ifs->tellg();
        int streamlineCount     = fileSize / (sizeof(float) * latDimMultiplier * model.latDim);
        totalCnt               += streamlineCount;
        ifs->seekg(0, std::ios::beg);  // Move the file pointer back to the beginning
        
        file.push_back(ifs);
        scnt.push_back(streamlineCount);
        scnt_cumSum.push_back(totalCnt);

        disp(MSG_DETAIL,"Including encoded streamlines from: %s", fname.c_str());
        return true;
    };

    if (isFile) { // If the input is a single file
        if(!addToProcess(inp_path)) return;
    } else { // If the input is a directory
        std::vector<std::string> binFiles = getMatchingFiles(inp_path + "/*.bin");
        for (auto& b : binFiles) {
            if(!addToProcess(b)) return;
        }
    }
    // ------

    if (totalCnt < batchSize) batchSize = totalCnt;

    // Compute number of streamlines to fetch from each file for each batch
    std::vector<int> scnt_batch;
    int tmpTotal = 0;
    for (auto& c : scnt) {
        float ratio = float(c) / float(totalCnt);
        int cnt = ratio * float(batchSize);
        if (cnt == 0) cnt = 1;
        tmpTotal += cnt;
        scnt_batch.push_back(cnt);
    }

    // Adjust the scnt_batch to ensure the total number is exactly batchSize
    int currentTotal = tmpTotal;
    while (currentTotal != batchSize) {
        if (currentTotal < batchSize) {
            // Add streamlines to the files
            for (size_t i = 0; i < scnt_batch.size(); ++i) {
                if (currentTotal == batchSize) break;
                if (scnt_batch[i] < scnt[i]) {  // Ensure we do not exceed the number of available streamlines
                    scnt_batch[i]++;
                    currentTotal++;
                }
            }
        } else if (currentTotal > batchSize) {
            // Remove streamlines from the files with higher counts
            for (size_t i = 0; i < scnt_batch.size(); ++i) {
                if (currentTotal == batchSize) break;
                if (scnt_batch[i] > 1) {  // Ensure that we keep at least one streamline per file
                    scnt_batch[i]--;
                    currentTotal--;
                }
            }
        }
    }
    // ------

    // Readers for latent representation of streamlines
    NIBR::RandomDoer rand;
    rand.init_uniform_int(totalCnt-1);

    std::function<std::vector<Eigen::VectorXf>(int)> getRandomBatch = [&](int readCnt) -> std::vector<Eigen::VectorXf> {

        int startInd = rand.uniform_int();

        // Find the file to start reading from
        int fileIndex = 0;
        while (fileIndex < int(scnt_cumSum.size()) && startInd >= scnt_cumSum[fileIndex]) {
            fileIndex++;
        }
        if (fileIndex == int(scnt_cumSum.size())) return std::vector<Eigen::VectorXf>();

        // Calculate local start index in the selected file
        int localStartInd = startInd - (fileIndex == 0 ? 0 : scnt_cumSum[fileIndex - 1]);

        // Determine how many streamlines we can read from this file
        int streamlinesToRead = std::min(readCnt, scnt[fileIndex] - localStartInd);

        // Create the output vector
        std::vector<Eigen::VectorXf> out;
        out.reserve(readCnt);

        // Read streamlines from the selected file
        std::vector<float> fileBatch(streamlinesToRead * latDimMultiplier * model.latDim);
        file[fileIndex]->seekg(localStartInd * sizeof(float) * latDimMultiplier * model.latDim, std::ios::beg);
        file[fileIndex]->read(reinterpret_cast<char*>(fileBatch.data()), sizeof(float) * latDimMultiplier * model.latDim * streamlinesToRead);

        for (int i = 0; i < streamlinesToRead; ++i) {
            Eigen::VectorXf eigenVec = Eigen::VectorXf::Map(fileBatch.data() + i * 2 * model.latDim, latDimMultiplier * model.latDim);
            out.push_back(std::move(eigenVec));
        }

        // If we could not read enough streamlines, read the remaining from another random file
        if (streamlinesToRead < readCnt) {
            int remaining = readCnt - streamlinesToRead;
            auto remainingBatch = getRandomBatch(remaining);  // Recursively read remaining batch

            if (int(remainingBatch.size()) != remaining) return std::vector<Eigen::VectorXf>();
            out.insert(out.end(), remainingBatch.begin(), remainingBatch.end());
        }

        return (int(out.size()) == readCnt) ? out : std::vector<Eigen::VectorXf>();
    };


    std::function<std::vector<Eigen::VectorXf>(int)> getRegularBatch = [&](int readCnt) -> std::vector<Eigen::VectorXf> {
        std::vector<Eigen::VectorXf> out;
        out.reserve(readCnt);

        for (size_t i = 0; i < scnt_batch.size(); ++i) {
            int streamlinesToRead = std::min(scnt_batch[i], readCnt - int(out.size()));
            if (streamlinesToRead <= 0) break;

            std::vector<float> fileBatch(streamlinesToRead * latDimMultiplier * model.latDim);
            file[i]->read(reinterpret_cast<char*>(fileBatch.data()), sizeof(float) * latDimMultiplier * model.latDim * streamlinesToRead);

            for (int s = 0; s < streamlinesToRead; s++) {
                Eigen::VectorXf eigenVec = Eigen::VectorXf::Map(fileBatch.data() + s * latDimMultiplier * model.latDim, latDimMultiplier * model.latDim);
                out.push_back(std::move(eigenVec));
            }
        }

        return out;
    };


    // Compute maxIteration if needed
    if (randomize) {
        if (maxIteration < 1) maxIteration = 1;
    } else {
        if (maxIteration < 1) maxIteration = std::ceil(float(totalCnt) / float(batchSize));    
    }

    // Do the clustering
    std::vector<Eigen::VectorXf> clusterCenters;
    
    // Build an empty KD-Tree
    PointCloud cloud;
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>,PointCloud,-1> KDTree;
    KDTree kdtree(model.latDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams( ((256/model.latDim) > 10) ? 10 : (256/model.latDim) ) ) ;
    kdtree.buildIndex();

    // Adjust maxDist based on model's distance scaling factor
    float adjMaxDist = (maxDist/model.distScaler) * (maxDist/model.distScaler);

    disp(MSG_INFO,"Clustering...");
    int totRead = 0;
    for (int iter = 0; iter < maxIteration; iter++) {
        
        // Get a batch
        std::vector<Eigen::VectorXf> batch;

        if (randomize) {
            int curTot = 0;
            while (curTot < batchSize) {
                disp(MSG_DETAIL,"Getting random batch of size %d ...", batchSize);
                auto miniBatch = getRandomBatch(batchSize - curTot);
                curTot += miniBatch.size();
                batch.insert(batch.end(), miniBatch.begin(), miniBatch.end());
                disp(MSG_DETAIL,"Done");
            }
        } else {
            disp(MSG_DETAIL,"Getting regular batch of size %d ...", batchSize);
            batch = getRegularBatch(batchSize);
            totRead += batch.size();
            if (totRead > totalCnt) batchSize = totalCnt - totRead;
            disp(MSG_DETAIL,"Done");
        }

        disp(MSG_DETAIL,"Current batchSize: %d", batchSize);

        // Shuffle the batch if needed
        if (shuffle) std::shuffle(batch.begin(), batch.end(), rand.getGen());


        // Perform clustering operations on the batch
        // This will be done in two steps
        // Step 1. If a streamline is far from all existing cluster centers, keep it as "unassigned".
        // Step 2. Within in batch, locally cluster all the "unassigned" streamlines, and form localClusterCenters.
        // Step 3. Append localClusterCenters to clusterCenters

        // For efficiency split the unassigned streamlines into smaller batches of:
        // 10000 streamlines in the first iteration when there will be many new clusters,
        // and 1000 streamlines in the other iteration when there will be many existing clusters, and less new clusters.

        std::vector<Eigen::VectorXf> localClusterCenters;       // New clusters found in this batch.
        
        // Build an empty KD-Tree for local clusters
        PointCloud localCloud;
        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>,PointCloud,-1> KDTree;
        KDTree localKdtree(model.latDim, localCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        localKdtree.buildIndex();

        std::vector<std::atomic<bool>> unassigned(batchSize);    // True if a streamline in the batch did not belong to any existing cluster
        for (int i = 0; i < batchSize; ++i) unassigned[i] = false;

        std::vector<size_t> unaInd;                              // Indices of the streamlines, which did not belong to any existing cluster
        std::vector<Eigen::VectorXf> unassignedClusterCenters;   // Clusters of the unassigned streamlines that were not clusered within a batch
        size_t taskOffset = 0;

        auto addToGlobalCluster = [&](NIBR::MT::TASK task) -> void {

            if (!clusterCenters.empty()) {
                size_t closestCenterIndex;
                float  squaredDistToClosestClusterCenter;

                nanoflann::KNNResultSet<float> resultSet1(1);
                resultSet1.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                kdtree.findNeighbors(resultSet1, batch[task.no].data(),             nanoflann::SearchParameters());
                if (squaredDistToClosestClusterCenter < adjMaxDist) return;

                if(newGenSize == 0) {
                    nanoflann::KNNResultSet<float> resultSet2(1);
                    resultSet2.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                    kdtree.findNeighbors(resultSet2, batch[task.no].data()+model.latDim, nanoflann::SearchParameters());
                    if (squaredDistToClosestClusterCenter < adjMaxDist) return;
                }
            }

            unassigned[task.no].store(true);

        };

        std::mutex mx;

        auto addToLocalCluster = [&](NIBR::MT::TASK task) -> void {

            size_t rInd = unaInd[task.no+taskOffset];

            if (!localClusterCenters.empty()) {
                size_t closestCenterIndex;
                float  squaredDistToClosestClusterCenter;

                nanoflann::KNNResultSet<float> resultSet1(1);
                resultSet1.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                localKdtree.findNeighbors(resultSet1, batch[rInd].data(),    nanoflann::SearchParameters());
                if (squaredDistToClosestClusterCenter < adjMaxDist) return;

                if(newGenSize == 0) {
                    nanoflann::KNNResultSet<float> resultSet2(1);
                    resultSet2.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                    localKdtree.findNeighbors(resultSet2, batch[rInd].data()+model.latDim, nanoflann::SearchParameters());
                    if (squaredDistToClosestClusterCenter < adjMaxDist) return;
                }
            }

            {
                mx.lock();

                for (size_t ind = 0; ind < unassignedClusterCenters.size(); ind++) {

                    float sum1 = 0;
                    float sum2 = 0;

                    for (int i = 0; i < model.latDim; i++) {
                        float d1 = (batch[rInd][i] - unassignedClusterCenters[ind][i]);
                        sum1    += d1 * d1;
                        if(newGenSize == 0) {
                            float d2 = (batch[rInd][i] - unassignedClusterCenters[ind][i + model.latDim]);
                            sum2    += d2 * d2;
                        }
                        
                    }

                    if (std::min(sum1,sum2) < adjMaxDist) {
                        mx.unlock();
                        return;
                    }
                }

                unassignedClusterCenters.push_back(batch[rInd]);
                mx.unlock();
                return;                
            }   

        };


        // Add the newly found localClusterCenters in the global clusterCenters
        auto doUnassigned = [&]() -> int {

            unaInd.clear();

            for (size_t r = 0; r < unassigned.size(); r++) {
                if (unassigned[r]) unaInd.push_back(r);
            }
            int unaCnt     = unaInd.size();

            int splitSize;
            if      (unaCnt <= 100000)  splitSize = 1000;
            else if (unaCnt <= 1000000) splitSize = 10000;
            else                        splitSize = 100000;

            taskOffset     = 0;
            int begInd     = 0;
            int endInd     = 0;
            int splitCnt   = (unaCnt / splitSize < 1) ? unaCnt : unaCnt/splitSize;
            int curSplitNo = 0;
            std::string preamble = "\033[1;32mNIBRARY::INFO: \033[0;32m";
            if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << preamble << "Clustering unassigned streamlines " << ": 0%" << "\033[0m" << '\r' << std::flush;}
            float progressScaler = 100.0f/float(splitCnt);                
            while (endInd != unaCnt) {
                begInd = endInd;
                endInd = ((begInd + splitSize) <= unaCnt) ? (begInd + splitSize) : unaCnt;
                int curSplitSize = endInd - begInd;
                NIBR::MT::MTRUN(curSplitSize, addToLocalCluster);
                localClusterCenters.insert(localClusterCenters.end(), unassignedClusterCenters.begin(), unassignedClusterCenters.end());
                localCloud.points = localClusterCenters;
                localKdtree.buildIndex();
                unassignedClusterCenters.clear();
                taskOffset += curSplitSize;
                if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << "\r\033[K" << std::flush;}
                if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << preamble << "Clustering unassigned streamlines: " << std::fixed << std::setprecision(2) << (++curSplitNo)*progressScaler << "%" << "\033[0m" << std::flush;}
            }
            if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << "\r\033[K" << preamble << "Clustering unassigned streamlines: 100%" << std::endl;}

            return unaCnt;

        }; 

        
        // Assign streamlines into existing clusters, and find "unassigned" streamlines, which were not assinged to any cluster
        NIBR::MT::MTRUN( batchSize, "Assigning clusters " + to_string_with_precision(iter+1,0) + " / " + to_string_with_precision(maxIteration,0), addToGlobalCluster);

        // Find localClusterCenters that are the cluster centers of the "unassigned" streamlines, 
        int unassignedCnt = doUnassigned();

        // Append the localClusterCenters to global clusterCenters
        clusterCenters.insert(clusterCenters.end(), localClusterCenters.begin(), localClusterCenters.end());
        
        // Update the global KD-tree
        cloud.points = clusterCenters;
        kdtree.buildIndex();

        disp(MSG_INFO,"Unassigned streamlines: %d - New clusters found: %d - Total clusters: %d", unassignedCnt, localClusterCenters.size(), clusterCenters.size());

    }
    disp(MSG_INFO,"Done");

    // Close all the files
    for (auto f : file) {
        f->close();
        delete f;
    }


    // Open binary file for writing the latent representation of the cluster centers
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        disp(MSG_ERROR,"Failed to open output file.");
        return;
    }
    
    for (const auto& c : clusterCenters) {
        ofs.write(reinterpret_cast<const char*>(c.data()), latDimMultiplier * model.latDim * sizeof(float));
    }
    ofs.close();


    return;
       
}          
    
     
void findClusterCenters(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("finds streamline cluster centers based on their latent space representation");

    app->add_option("<input>",               inp_path,           "Input path. Either a file (.bin) or a folder.")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output latent representations of cluster centers (.clc).")
        ->required();

    app->add_option("--model,-m",            model_spec,         "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. The distance scaler factor of the model can be obtained using the \"modelTest\" command.");

    app->add_option("--maxDist, -d",         maxDist,            "Maximum distance from any cluster center.")
        ->required();

    app->add_option("--maxIteration, -i",    maxIteration,       "Limits maximum number of iterations in constrast to the default, which iterates until all streamlines processed.");
    
    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Number of streamlines to process at each iteration. Default: 1000000");
    app->add_flag("--shuffle, -s",           shuffle,            "Shuffles streamlines within batches that leads to different cluster centers at each run.");
    
    app->add_flag("--randomize, -r",         randomize,          "Cluster using randomized batches instead of the default, regularly fetched, batches.");
    app->add_option("--miniBatchSize",       miniBatchSize,      "When using random batches, each batch is split into mini batches fetched contigously from a single file, miniBatchSize sets that value. Default: 1000");

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--newGenSize",           newGenSize,       "Amount of non-flipped values in new gen models");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_findClusterCenters);  
     
}

                       