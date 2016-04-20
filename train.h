#ifndef TRAIN_H
#define TRAIN_H

void read_problem(const char *filename);
int train_main(int argc, char **argv);
//extern int predict_main(int,char**);

extern struct parameter param;
extern struct problem prob;
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);

#endif //TRAIN_H