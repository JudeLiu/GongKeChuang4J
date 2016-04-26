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
#include <unordered_map>
#include <iostream>
#include <errno.h>
#include "linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

typedef long unsigned int size_type;

void parse_command_line(int, char **, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void print_null(const char *s);
void exit_with_help();
static char* readline(FILE *input);
void do_cross_validation();
void do_find_parameter_C();
void exit_input_error(int line_num);
void priori_min_max_train(int, char**);
void __priori_min_max_train(char*);
void random_min_max_train(int argc, char** argv);
void __random_min_max_train(char* test_file_name);
void naive_train(int, char**);
void __naive_train(char*);
std::string transformLabel(std::string);
std::vector<std::string> split(const std::string&, char);
std::string changeFileName(const std::string&);

struct feature_node *x_space;
struct parameter param={};
struct problem prob={};
int flag_cross_validation;
int flag_find_C;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;

#endif //HEAD_H