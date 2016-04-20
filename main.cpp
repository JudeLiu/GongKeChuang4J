#include <iostream>
#include <cstring>
#include <cstdlib>
#include <errno.h>
#include "linear.h"
#include "train.h"


//extern int train_main(int,char**);
//extern int predict_main(int,char**);

//extern void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
//extern void read_problem(const char *filename);

int main(int argc, char **argv)
{
    //char *modelFileName = NULL;
    //sprintf(modelFileName,"model");
    char input_file_name[1024];
    char model_file_name[1024];
    parse_command_line(argc, argv, input_file_name, model_file_name);
    read_problem(input_file_name);
    std::cout<<"start train\n\n";
    //Train::train_main(argc,argv);
    struct model* _model = train(&prob,&param);
    if(save_model(model_file_name,_model))
    {
        std::cerr<<"can't save model to file "<<model_file_name<<std::endl;
        exit(1);
    }
    free_and_destroy_model(&_model);
        destroy_param(&param);
    free(prob.y);
    free(prob.x);
    //free(x_space);
    //free(line);
    std::cout<<"training finished\n\nstart predicting\n";
/*
    char** argin = new char*[3];
    sprintf(argin[0],"data/myTest.txt");
    sprintf(argin[1],"model");
    sprintf(argin[2], "predict_out");
    Predict::predict_main(4,argin);
    delete []argin;
    printf("predicting finished\n");
    */
    return 0;
}
