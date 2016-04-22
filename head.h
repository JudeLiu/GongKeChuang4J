#ifndef HEAD_H
#define HEAD_H

#include <ctime>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <vector>
#include <set>
#include <iostream>
#include <errno.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

typedef long unsigned int size_type;

void transformLabel(char*);
void parse_command_line(int, char **, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void print_null(const char *s);
void exit_with_help();
static char* readline(FILE *input);
void do_cross_validation();
void do_find_parameter_C();
void exit_input_error(int line_num);
void min_max_train(int argc, char** argv);
void __min_max_train(char* test_file_name);

struct feature_node *x_space;
struct parameter param={};
struct problem prob={};
struct model* model_;
int flag_cross_validation;
int flag_find_C;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;

#endif //HEAD_H