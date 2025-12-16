#include "cmd.h"


void run_ae_dummy()
{
    disp(MSG_ERROR, "Autoencoder support is not compiled.");
    return;
}

void modelTest(CLI::App* app) 
{  
    app->description("Autoencoder support is not compiled");
    app->callback(run_ae_dummy);
} 

void encode(CLI::App* app) 
{  
    app->description("Autoencoder support is not compiled");
    app->callback(run_ae_dummy);
} 

void decode(CLI::App* app) 
{  
    app->description("Autoencoder support is not compiled");
    app->callback(run_ae_dummy);
} 