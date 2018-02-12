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

#define alpha 0.005
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
double P0,P1,P2,P3,P4;

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
	if(func == 1)return P0 + P1 * d.sqft  + P2 * d.floors  + P3 * d.bedrooms + P4 * d.bathrooms;
	if(func == 2)return P0 + P1 * d.sqft * d.sqft + P2 * d.floors * d.floors + P3 * d.bedrooms * d.bedrooms  * d.bedrooms + P4 * d.bathrooms * d.bathrooms * d.bathrooms;
	if(func ==3)return P0 + P1 * d.sqft * d.sqft * d.sqft + P2 * d.floors * d.floors * d.floors + P3 * d.bedrooms * d.bedrooms + P4 * d.bathrooms * d.bathrooms;
}
void normalize()
{
	double sqftmean = 0,sqftmax = -1,floormean = 0,floormax = -1,bedmean = 0,bedmax = -1,bathmean = 0,bathmax = -1;
	for(int i = 0; i < DATASET.size(); i++)
	{
		sqftmean += DATASET[i].sqft;
		floormean += DATASET[i].floors;
		bedmean += DATASET[i].bedrooms;
		bathmean += DATASET[i].bathrooms;
	}
	sqftmean /= DATASET.size();
	floormean /= DATASET.size();
	bedmean /= DATASET.size();
	bathmean /= DATASET.size();

	for(int i = 0; i < DATASET.size(); i++)
	{
		if(DATASET[i].sqft > sqftmax) sqftmax = DATASET[i].sqft;
		if(DATASET[i].floors > floormax) floormax = DATASET[i].floors;
		if(DATASET[i].bedrooms > bedmax) bedmax = DATASET[i].bedrooms;
		if(DATASET[i].bathrooms > bathmax) bathmax = DATASET[i].bathrooms;
	}

	for(int i = 0; i < DATASET.size(); i++)
	{
		DATASET[i].sqft = (DATASET[i].sqft )/sqftmax;
		DATASET[i].floors = (DATASET[i].floors )/floormax;
		DATASET[i].bedrooms = (DATASET[i].bedrooms)/bedmax;
		DATASET[i].bathrooms = (DATASET[i].bathrooms)/bathmax;
	}

}

void initialize()
{
	P0 = 0;
	P1 = P2 = P3 = P4 = 0; 
}

void gradientDescent(int func)
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	double sum0 = 0,sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	while(converge >= 0.01)
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
		//cout << "Jsum : " << Jsum << endl;
		//cout << "oldJ : " << oldJ << endl;
		//cout << "converge : " << converge << endl;
		oldJ = newJ;

		sum0 = 0;sum1 = 0;sum2 = 0;sum3 = 0;sum4 = 0;

		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			sum0 += error[i]/ TRAININGSET.size();
			sum1 += error[i] * (TRAININGSET[i].sqft)/ TRAININGSET.size();
			sum2 += error[i] * (TRAININGSET[i].floors)/ TRAININGSET.size();
			sum3 += error[i] * (TRAININGSET[i].bedrooms)/ TRAININGSET.size();
			sum4 += error[i] * (TRAININGSET[i].bathrooms)/ TRAININGSET.size();
		}
		
		P0 = P0 - (alpha * sum0);
		P1 = P1 - (alpha * sum1);
		P2 = P2 - (alpha * sum2);
		P3 = P3 - (alpha * sum3);
		P4 = P4 - (alpha * sum4);

		count ++;
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
	ifstream myfile("data.csv");
	string value;
	vector<double> tokens;
	int rows = 0;
	data d;
	getline (myfile, value, '\n' );
	while(myfile.good())
	{
	     getline (myfile, value, '\n' );
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
	normalize();
	for(i = 0 ; i < 0.8 * rows; i++)
	{
		TRAININGSET.push_back(DATASET[i]);
	}

	for(int j = i; j < DATASET.size(); j++)
	{
		TESTSET.push_back(DATASET[j]);
		//TESTSET[TESTSET.size() - 1].price = 0;
	}

	initialize();
	gradientDescent(1);

	// for(i = 0; i < TRAININGSET.size(); i++)
	// {
	// 	cout << TRAININGSET[i].sqft << " " << TRAININGSET[i].floors << " " << TRAININGSET[i].bedrooms<< " " << TRAININGSET[i].bathrooms << " " << TRAININGSET[i].price << endl;
	// }

	cout << P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;

	for(int j = 0; j < TESTSET.size(); j++)
	{
		cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],1) << endl; 
	}
	cout << endl;
	cout << "error for linear case is : " << errorCalc(1) << endl;

	initialize();
	gradientDescent(2);

	cout << P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;

	for(int j = 0; j < TESTSET.size(); j++)
	{
		cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],2) << endl; 
	}
	cout << endl;
	cout << "error for quadratic case is : " << errorCalc(2) << endl;

	initialize();
	gradientDescent(3);

	cout << P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;

	for(int j = 0; j < TESTSET.size(); j++)
	{
		cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j],3) << endl; 
	}
	cout << endl;
	cout << "error for cubic case is : " << errorCalc(3) << endl;


	return 0;
}
