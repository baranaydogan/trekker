#include "cmd.h"
#include "dMRI/tractography/autoencoder/StreamlineAutoencoder.h"
#include "dMRI/tractography/autoencoder/decodeAndSave.h"

using namespace NIBR;

namespace CMDARGS_DECODE {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string> model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, datatype

    int  batchSize          = 512;
    bool useCPU             = false;
    bool skip               = false;
    bool tck                = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    size_t newGenSize       = 0;
}

using namespace CMDARGS_DECODE; 
 
void run_decode()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    bool isFile = false;
    
    std::string ext = getFileExtension(inp_path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "bin" || ext == "clc")
        isFile = true;

    bool isFolder = existsFolder(inp_path);

    if (!isFile && !isFolder) {
        disp(MSG_ERROR,"Unknown input");
        return;
    }

    // Set model
    NIBR::StreamlineAutoencoder model = NIBR::StreamlineAutoencoder(model_spec, useCPU, newGenSize);
    if (!model.isReady()) return;

    if (isFile) {
        decodeAndSave(inp_path,out_path,!skip,model,batchSize);
        return;
    }

    
    if(!makeFolder(out_path)) return;

    std::vector<std::string> binFiles = getMatchingFiles(inp_path + "/*.bin");
    std::vector<std::string> clcFiles = getMatchingFiles(inp_path + "/*.clc");

    std::vector<std::string> files = binFiles;
    files.insert(files.end(),clcFiles.begin(),clcFiles.end());

    for (auto& t : files) {
        auto out = (tck) ? out_path+"/"+removeFileExtension(t)+".tck" : out_path+"/"+removeFileExtension(t)+".vtk";
        if (!skip && !parseForceOutput(out,force)) return;
    }

    for (auto& t : files) {
        std::filesystem::path p(t);

        auto out = (tck) ? out_path+"/"+removeFileExtension(t)+".tck" : out_path+"/"+removeFileExtension(t)+".vtk";

        if (existsFile(out) && skip) {
            disp(MSG_INFO, "Skipping %s -> %s", p.filename().c_str(),out.c_str());    
        } else {
            disp(MSG_INFO, "Processing %s -> %s", p.filename().c_str(),out.c_str());
            decodeAndSave(t,out,!skip,model,batchSize);
        }
    }

    return;
       
}          
    
     
void decode(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("decodes latent space representations and output a tractogram");

    app->add_option("<input>",               inp_path,           "Input path. Either latent space representations of streamlines (.bin) or a folder containing laten representations.")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output path. Either a tractogram file (.vtk,.tck) or a folder.")
        ->required();

    app->add_option("--model,-m",            model_spec,         "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions and data type (float or double). E.g. /model/test_model.pt 256 64 float");

    app->add_option("--newGenSize",           newGenSize,       "Amount of non-flipped values in new gen models");
    
    app->add_flag("--skip, -s",              skip,               "Skip if the output exists.");
    app->add_flag("--tck, -t",               tck,                "Write .tck files instead of the default .vtk");

    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Default: 512.");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_decode);  
     
}

                       