#include "cmd.h"
#include "dMRI/tractography/autoencoder/StreamlineAutoencoder.h"
#include "dMRI/tractography/autoencoder/autoEncoderUtils.h"

using namespace NIBR;

namespace CMDARGS_ENCODE {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string> model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, datatype

    int  batchSize          = 512;
    bool useCPU             = false;
    bool skip               = false;
    
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    size_t newGenSize       = 0;
}

using namespace CMDARGS_ENCODE; 

void run_encode()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    bool isFile = false;
    
    std::string ext = getFileExtension(inp_path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "vtk" || ext == "tck")
        isFile = true;

    bool isFolder = existsFolder(inp_path);

    if (!isFile && !isFolder) {
        disp(MSG_ERROR,"Unknown input");
        return;
    }

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(model_spec, useCPU, newGenSize);
    if (!model.isReady()) return;

    if (isFile) {
        encodeAndSave(inp_path,out_path,!skip,model,batchSize);
        return;
    }

    
    if(!makeFolder(out_path)) return;

    std::vector<std::string> tckFiles = getMatchingFiles(inp_path + "/*.tck");
    std::vector<std::string> vtkFiles = getMatchingFiles(inp_path + "/*.vtk");

    std::vector<std::string> tractogramFiles = tckFiles;
    tractogramFiles.insert(tractogramFiles.end(),vtkFiles.begin(),vtkFiles.end());

    for (auto& t : tractogramFiles) {
        auto out = out_path+"/"+removeFileExtension(t)+".bin";
        if (!skip && !parseForceOutput(out,force)) return;
    }

    for (auto& t : tractogramFiles) {
        std::filesystem::path p(t);

        auto out = out_path+"/"+removeFileExtension(t)+".bin";

        if (existsFile(out) && skip) {
            disp(MSG_INFO, "Skipping %s -> %s", p.filename().c_str(),out.c_str());    
        } else {
            disp(MSG_INFO, "Processing %s -> %s", p.filename().c_str(),out.c_str());
            encodeAndSave(t,out,!skip,model,batchSize);
        }
    }

    return;
       
}          
    
     
void encode(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "A streamline is representated in a latent space. We also flip the streamline and compute the latent representation for the flipped version. Overall, a streamline is encoded with 2 x latent dimension. First latent dimension values encode encodes the original order of points, the latter values encode the flipped streamline.";

    setInfo(app,info);

    app->description("encodes tractograms and saves their latent space representation");

    app->add_option("<input>",               inp_path,           "Input path. Either a tractogram (.vtk, .tck) or a folder containing tractograms.")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output path. Either a file (.bin) or a folder.")
        ->required();

    app->add_option("--model,-m",            model_spec,         "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions and data type (float or double). E.g. /model/test_model.pt 256 64 float");

    app->add_flag("--skip, -s",              skip,               "Skip if the output exists.");

    app->add_option("--newGenSize",           newGenSize,       "Amount of non-flipped values in new gen models");

    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Default: 512.");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_encode);  
     
}
