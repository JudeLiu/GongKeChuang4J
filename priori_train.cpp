#include "linear.h"
#include "head.h"

using namespace std;

static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void print_null(const char *s) {}

void exit_with_help()
{
    printf(
    "Usage: train [options] training_set_file [model_file]\n"
    "options:\n"
    "-s type : set type of solver (default 1)\n"
    "  for multi-class classification\n"
    "    0 -- L2-regularized logistic regression (primal)\n"
    "    1 -- L2-regularized L2-loss support vector classification (dual)\n"
    "    2 -- L2-regularized L2-loss support vector classification (primal)\n"
    "    3 -- L2-regularized L1-loss support vector classification (dual)\n"
    "    4 -- support vector classification by Crammer and Singer\n"
    "    5 -- L1-regularized L2-loss support vector classification\n"
    "    6 -- L1-regularized logistic regression\n"
    "    7 -- L2-regularized logistic regression (dual)\n"
    "  for regression\n"
    "   11 -- L2-regularized L2-loss support vector regression (primal)\n"
    "   12 -- L2-regularized L2-loss support vector regression (dual)\n"
    "   13 -- L2-regularized L1-loss support vector regression (dual)\n"
    "-c cost : set the parameter C (default 1)\n"
    "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
    "-e epsilon : set tolerance of termination criterion\n"
    "   -s 0 and 2\n"
    "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
    "       where f is the primal function and pos/neg are # of\n"
    "       positive/negative data (default 0.01)\n"
    "   -s 11\n"
    "       |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
    "   -s 1, 3, 4, and 7\n"
    "       Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
    "   -s 5 and 6\n"
    "       |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
    "       where f is the primal function (default 0.01)\n"
    "   -s 12 and 13\n"
    "       |f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
    "       where f is the dual function (default 0.1)\n"
    "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
    "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
    "-v n: n-fold cross validation mode\n"
    "-C : find parameter C (only for -s 0 and 2)\n"
    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

int train_main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	if (flag_find_C)
	{
		do_find_parameter_C();
	}
	else if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model_=train(&prob, &param);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

void do_find_parameter_C()
{
	double start_C, best_C, best_rate;
	double max_C = 1024;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;
	printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
	find_parameter_C(&prob, &param, nr_fold, start_C, max_C, &best_C, &best_rate);
	printf("Best C = %g  CV accuracy = %g%%\n", best_C, 100.0*best_rate);
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			  );
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}

//void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.init_sol = NULL;
	flag_cross_validation = 0;
	flag_C_specified = 0;
	flag_solver_specified = 0;
	flag_find_C = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				flag_solver_specified = 1;
				break;

			case 'c':
				param.C = atof(argv[i]);
				flag_C_specified = 1;
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'C':
				flag_find_C = 1;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		//strcpy(model_file_name,argv[i+1]);
		strcpy(test_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(test_file_name,"%s.model",p);
	}

	// default solver for parameter selection is L2R_L2LOSS_SVC
	if(flag_find_C)
	{
		if(!flag_cross_validation)
			nr_fold = 5;
		if(!flag_solver_specified)
		{
			fprintf(stderr, "Solver not specified. Using -s 2\n");
			param.solver_type = L2R_L2LOSS_SVC;
		}
		else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC)
		{
			fprintf(stderr, "Warm-start parameter search only available for -s 0 and -s 2\n");
			exit_with_help();
		}
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

/* 
 * generate random sequence from 0 to range-1, without repitiion, covering all
 * numbers from 0 to range-1
 */
vector<int> genRandomIndex(unsigned int range)
{
	vector<int> res;
	vector<bool> exist(range,false);
	while(res.size()<range)
	{
		int t = rand()%range;
		if(!exist[t])
		{
			res.push_back(t);
			exist[t] = true;
		}		
	}
	return res;
}

/*
 * decompose origin training set into 6 groups, 3 A, 3 not A
 * thus totally 9 subproblems
 */
void __min_max_train(char* test_file_name)
{
	const int subprobNo = 3;
	//decompose
	cout<<"start decomposing\n";
		srand(time(NULL));

	//find which are A, which are not A
	vector<int> groupA;
	vector<int> groupNotA;
	for(int i=0;i<prob.l;i++)
	{
		if(prob.y[i] == 1)
			groupA.push_back(i);
		else
			groupNotA.push_back(i);
	}

	//generate random sequence to split origin problem
	vector<int> randomIdxOfA = genRandomIndex(groupA.size());
	vector<int> randomIdxOfNA = genRandomIndex(groupNotA.size());

	//
	vector<problem> sub_problems(subprobNo*subprobNo);

	int spIdx = 0;
	int l = prob.l/subprobNo;
	for(int idxA=0;idxA<subprobNo;idxA++)
	{
		for(int idxNA=0;idxNA<subprobNo;idxNA++)
		{
			sub_problems[spIdx].l = l;
			sub_problems[spIdx].bias = -1;
			sub_problems[spIdx].x = new feature_node*[l];
			sub_problems[spIdx].y = new double[l];

			int sa = groupA.size()/subprobNo;
			int i=0;

			//copy 1/3 groupA and groupNA to sub_problems[spIdx] in each loop
			for(i=0;i<sa;i++)
			{
				int subscript = groupA[randomIdxOfA[i+idxA*sa]];
				sub_problems[spIdx].x[i] = prob.x[subscript]; //copy of pointer
				sub_problems[spIdx].y[i] = prob.y[subscript];
			}
			int sna = groupNotA.size()/subprobNo;
			for(;i<l;i++)
			{
				int subscript = groupNotA[randomIdxOfNA[i-sa+idxNA*sna]];
				sub_problems[spIdx].x[i] = prob.x[subscript];
				sub_problems[spIdx].y[i] = prob.y[subscript];
			}
			spIdx++;
		}
	}
	
	//train subproblem seperately
	cout<<"start training subproblem\n";
	clock_t start = clock(),stop,total;
	vector<model*> sub_models(subprobNo*subprobNo);
	for(int i=0;i<subprobNo*subprobNo;i++)
	{
		sub_models[i] = train(&sub_problems[i],&param);
	}
	stop = clock();
	total = stop-start;
	cout<<"subproblem training cost: "<<total<<endl<<endl;

	// read test set
	read_problem(test_file_name);

	/*
	 * min prodecure of min-max, predict individually and vote
	 */
	cout<<"voting start\n";
	start = clock();
	vector<vector<int>> pred_vote(subprobNo*subprobNo);
	for(int i=0;i<subprobNo*subprobNo;i++)
		for(int k=0;k<prob.l;k++)
			pred_vote[i].push_back(predict(
											sub_models[i],
											prob.x[k]));
	stop = clock();
	total += stop - start;

	/*
	 * count vote and do MIN
	 */
	cout<<"start MIN\n";
	vector<vector<int>> after_min(subprobNo);
	spIdx = 0;
	start = clock();
	for(int idxA=0;idxA<subprobNo;idxA++)
	{
		//initialize
		for(int k=0;k<prob.l;k++)
			after_min[idxA].push_back(0);
		for(int idxNA=0;idxNA<subprobNo;idxNA++)
		{
			//count vote
			for(int k=0;k<prob.l;k++)
				after_min[idxA][k] += pred_vote[spIdx][k];
			spIdx++;
		}
		//MIN, if all predict i, the predict 1; otherwise predict 0
		for(int k=0;k<prob.l;k++)
			if(after_min[idxA][k] == subprobNo)
				after_min[idxA][k] = 1;
			else
				after_min[idxA][k] = 0;
	}
	stop = clock();
	total += stop - start;
	cout<<"MIN cost: "<<(stop-start)<<endl<<endl;

	/* 
	 * do MAX, if one predict 1, then predict 1; otherwise predict 0
	 */
	cout<<"start MAX\n";
	vector<int> after_max(prob.l,0);
	start = clock();
	for(int i=0;i<prob.l;i++)
	{
		for(auto minIter : after_min)
		{
			after_max[i] += minIter[i];
		}
		if(after_max[i]>0)
			after_max[i] = 1;
		else
			after_max[i] = 0;
	}
	stop = clock();
	total += stop - start;
	cout<<"MAX cost: "<<(stop - start)<<endl<<endl;


	/*
	 * compute accuracy
	 */
	int cnt=0;
	for(int i=0;i<prob.l;i++)
	
		if(after_max[i]==prob.y[i]) cnt++;
	cout<<"total time: "<<(float)total/CLOCKS_PER_SEC<<'s'<<endl;
	cout<<"accuracy = "<<(cnt*1.0/prob.l * 100)<<"\%\n";
}

void min_max_train(int argc, char** argv)
{
	char input_file_name[1024];
	//char model_file_name[1024];
	char test_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, test_file_name);
	read_problem(input_file_name);
	error_msg = check_parameter(&prob,&param);

	__min_max_train(test_file_name);

	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	
	/*
	if (flag_find_C)
	{
		do_find_parameter_C();
	}
	else if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model_=train(&prob, &param);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	free(line);
*/
}

int main(int argc, char** argv)
{
	min_max_train(argc,argv);
}