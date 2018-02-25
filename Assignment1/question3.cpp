#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <limits>
#include <cfloat>

using namespace std;

//#define alpha 0.005
double alpha;
#define lambda 1

typedef struct DATA
{
	double sqft;
	double floors;
	double bedrooms;
	double bathrooms;
	double price;
}data;

vector<data> DATASET;
vector<data> TRAININGSET;
vector<data> TESTSET;
vector<double> Params;

vector<double> parse(string s)
{
	string delimiter = ",";
	vector<double> parsed;
	
	size_t pos = 0;
	string token;
	double a;
	while ((pos = s.find(delimiter)) != string::npos) {
	    token = s.substr(0, pos);
	    stringstream tok;
	    tok.str(token);
	    tok >> a;
	    parsed.push_back(a);
	    s.erase(0, pos + delimiter.length());
	}
	stringstream tok;
	tok.str(s);
	tok >> a;
	parsed.push_back(a);

	return parsed;
}

double hypothesisFuncion(data d,int func)
{
	if(func == 1)
		return Params[0] + Params[1] * d.sqft  + Params[2] * d.floors  + Params[3] * d.bedrooms + Params[4] * d.bathrooms;
	if(func == 2)
		return  hypothesisFuncion(d,1)+ Params[5] * (pow(d.sqft,2)) + Params[6] * (pow(d.floors,2)) + Params[7] * (pow(d.bedrooms,2))  + Params[8]* (pow(d.bathrooms,2));
	if(func ==3)
		return  hypothesisFuncion(d,2)+ Params[9] * (pow(d.sqft,3)) + Params[10] * (pow(d.floors,3)) + Params[11] * (pow(d.bedrooms,3))  + Params[12]* (pow(d.bathrooms,3));
}
double deviation (vector<double> x, int mean)
{
	double deviation;
	double sum2;

	for ( int i = 0; i < x.size(); i++ )
	{
		sum2 += ((x[i]-mean) * (x[i]-mean)) ;
	}
	deviation= sqrt(sum2/(x.size()));
	return deviation;
} 


void normalize()
{
	double sqftmean = 0,sqftmax = -1,floormean = 0,floormax = -1,bedmean = 0,bedmax = -1,bathmean = 0,bathmax = -1;
	double devsqft,devfloors,devbed,devbath;
	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		sqftmean += TRAININGSET[i].sqft;
		floormean += TRAININGSET[i].floors;
		bedmean += TRAININGSET[i].bedrooms;
		bathmean += TRAININGSET[i].bathrooms;
	}
	sqftmean /= TRAININGSET.size();
	floormean /= TRAININGSET.size();
	bedmean /= TRAININGSET.size();
	bathmean /= TRAININGSET.size();

	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		if(TRAININGSET[i].sqft > sqftmax) sqftmax = TRAININGSET[i].sqft;
		if(TRAININGSET[i].floors > floormax) floormax = TRAININGSET[i].floors;
		if(TRAININGSET[i].bedrooms > bedmax) bedmax = TRAININGSET[i].bedrooms;
		if(TRAININGSET[i].bathrooms > bathmax) bathmax = TRAININGSET[i].bathrooms;
	}

	vector<double> x;
	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		x.push_back(TRAININGSET[i].sqft);
	}
	devsqft = deviation(x,sqftmean); x.clear();

	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		x.push_back(TRAININGSET[i].floors);
	}
	devfloors = deviation(x,floormean); x.clear();

	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		x.push_back(TRAININGSET[i].bedrooms);
	}
	devbed = deviation(x,bedmean); x.clear();

	for(int i = 0; i < TRAININGSET.size(); i++)
	{
		x.push_back(TRAININGSET[i].bathrooms);
	}
	devbath = deviation(x,bathmean); x.clear();


	for(int i = 0; i < DATASET.size(); i++)
	{
		DATASET[i].sqft = (DATASET[i].sqft - sqftmean)/devsqft;
		DATASET[i].floors = (DATASET[i].floors - floormean)/devfloors;
		DATASET[i].bedrooms = (DATASET[i].bedrooms - bedmean)/devbed;
		DATASET[i].bathrooms = (DATASET[i].bathrooms - bathmean)/devbath;
	}

}

void initialize()
{
	for(int i = 0 ; i < 13;i++)
	{
		Params.push_back(static_cast <float> (rand()) / static_cast <float> (RAND_MAX)); 
	}
}

void gradientDescent(int func)
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	vector<double> sum;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	//while(converge >= 0.01)
	while(++count)
	{
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			h = hypothesisFuncion(TRAININGSET[i],func);
			y = TRAININGSET[i].price;
			error[i] = h - y;
		}


		Jsum = 0;
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			Jsum += (error[i] * error[i])/(2*TRAININGSET.size());
		}

		newJ =  Jsum ;
		converge = oldJ - newJ;
		if(count == 1 || count == 100 || count == 200 || count == 500 || count == 1000 || count == 1500 || count == 2000 || count == 5000 || count % 10000 == 0)
		{
			cout << "error after " << count << " iterations : " << Jsum << "   ";
			cout << "convergence : " << converge << endl;

		}
		//cout << "Jsum : " << Jsum << endl;
		//cout << "oldJ : " << oldJ << endl;
		//cout << "converge : " << converge << endl;
		oldJ = newJ;

		for(int  i = 0; i < 13; i++)
		{
			sum.push_back(0);
		}

		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			sum[0] += error[i]/ TRAININGSET.size();
			sum[1] += error[i] * (TRAININGSET[i].sqft)/ TRAININGSET.size();
			sum[2] += error[i] * (TRAININGSET[i].floors)/ TRAININGSET.size();
			sum[3] += error[i] * (TRAININGSET[i].bedrooms)/ TRAININGSET.size();
			sum[4] += error[i] * (TRAININGSET[i].bathrooms)/ TRAININGSET.size();

			sum[5] += error[i] * (pow(TRAININGSET[i].sqft,2))/ TRAININGSET.size();
			sum[6] += error[i] * (pow(TRAININGSET[i].floors,2))/ TRAININGSET.size();
			sum[7] += error[i] * (pow(TRAININGSET[i].bedrooms,2))/ TRAININGSET.size();
			sum[8] += error[i] * (pow(TRAININGSET[i].bathrooms,2))/ TRAININGSET.size();

			sum[9] += error[i] * (pow(TRAININGSET[i].sqft,3))/ TRAININGSET.size();
			sum[10] += error[i] * (pow(TRAININGSET[i].floors,3))/ TRAININGSET.size();
			sum[11] += error[i] * (pow(TRAININGSET[i].bedrooms,3))/ TRAININGSET.size();
			sum[12] += error[i] * (pow(TRAININGSET[i].bathrooms,3))/ TRAININGSET.size();

		}
		
		for(int  i = 0; i < 13; i++)
		{
			Params[i] -= (alpha * sum[i]);
		}
		
		sum.clear();
		if((count == 10000 && func == 1) || (count == 10000 && func == 2) || (count == 10000 && func == 3) )break;
	}
	cout << count << endl;
}

double errorCalc(int func)
{
	double tot = 0;
	for(int i = 0; i < TESTSET.size(); i++)
	{
		tot += (TESTSET[i].price - hypothesisFuncion(TESTSET[i],func)) * (TESTSET[i].price - hypothesisFuncion(TESTSET[i],func));
	}
	tot = tot/TESTSET.size();
	return sqrt(tot);
}

int main()
{
	ofstream myfile;
	ifstream inputfile("data.csv");
	string value;
	vector<double> tokens;
	int rows = 0;
	data d;
	getline (inputfile, value, '\n' );
	while(inputfile.good())
	{
	     getline (inputfile, value, '\n' );
	     tokens = parse(value);
	    
	     d.sqft = tokens[0];
	     d.floors = tokens[1];
	     d.bedrooms = tokens[2];
	     d.bathrooms = tokens[3];
	     d.price = tokens[4];
	     DATASET.push_back(d);
	     tokens.clear();
	     rows++;
	}
	int i;
	for(i = 0 ; i < 0.8 * rows; i++)
	{
		TRAININGSET.push_back(DATASET[i]);
	}

	for(int j = i; j < DATASET.size(); j++)
	{
		TESTSET.push_back(DATASET[j]);
		//TESTSET[TESTSET.size() - 1].price = 0;
	}
	normalize();
	TRAININGSET.clear();
	TESTSET.clear();
	for(i = 0 ; i < 0.8 * rows; i++)
	{
		TRAININGSET.push_back(DATASET[i]);
	}

	for(int j = i; j < DATASET.size(); j++)
	{
		TESTSET.push_back(DATASET[j]);
		//TESTSET[TESTSET.size() - 1].price = 0;
	}
	myfile.open("pyinput.txt", std::ios::out | std::ios::app);
	alpha = 1;
	for(int j = 0; j < 10; j++)
	{
		alpha /= 2;	
		myfile << alpha << " ";
	}
	myfile << "\n";
	alpha = 1;
	for(int j = 0; j < 10; j++)
	{
		alpha /= 2;	
		cout << "linear case with alpha = " << alpha << " ..." << endl; 
		initialize();
		gradientDescent(1);
		for(int  i = 0; i < 5; i++)
		{
			cout << Params[i] << " ";
		}
		cout << endl;

		// for(int j = 0; j < TESTSET.size(); j++)
		// {
		// 	cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],1) << endl; 
		// }
		//cout << endl;
		cout << "error for linear case when alpha = " << alpha << " is " << errorCalc(1) << endl;
		myfile << errorCalc(1) << " ";
		cout << endl;
		Params.clear();
	}
	myfile << "\n";

	alpha = 0.005;
	for(int j = 0; j < 10; j++)
	{
		alpha /= 2;	
		myfile << alpha << " ";
	}
	myfile << "\n";
	alpha = 0.005;
	for(int j = 0; j < 10; j++)
	{
		alpha /= 2;	
		cout << "quadratic case with alpha = " << alpha << " ..." << endl; 
		initialize();
		gradientDescent(2);

		for(int  i = 0; i < 9; i++)
		{
			cout << Params[i] << " ";
		}
		cout << endl;

		// for(int j = 0; j < TESTSET.size(); j++)
		// {
		// 	cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],2) << endl; 
		// }
		
		cout << "error for quadratic case when alpha = " << alpha << " is " << errorCalc(2) << endl;
		myfile << errorCalc(2) << " ";
		cout << endl;
		Params.clear();
	}
	myfile << "\n";

	alpha = 0.000005;
	for(int j = 0; j < 10; j++)
	{
		alpha /= 2;	
		myfile << alpha << " ";
	}
	myfile << "\n";
	alpha = 0.000005;
	for(int j = 0; j < 10; j++)
	{

		alpha /= 2;	
		cout << "cubic case with alpha = " << alpha << " ..." << endl; 
		initialize();
		gradientDescent(3);

		for(int  i = 0; i < 13; i++)
		{
			cout << Params[i] << " ";
		}
		cout << endl;

		// for(int j = 0; j < TESTSET.size(); j++)
		// {
		// 	cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],3) << endl; 
		// }
		//cout << endl;
		cout << "error for cubic case when alpha = " << alpha << " is " << errorCalc(3) << endl;
		myfile << errorCalc(3) << " ";
		cout << endl;
		Params.clear();
	}
	myfile << "\n";
	myfile.close();
	
	return 0;
}
