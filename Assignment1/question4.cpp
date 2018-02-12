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

#define alphacube 0.0000007 //for cubic case
#define alpha 0.008
#define lambda 5

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

double hypothesisFuncion(data d)
{
	return P0 + P1 * d.sqft + P2 * d.floors + P3 * d.bedrooms + P4 * d.bathrooms;
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
	P0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	P1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    P2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	P3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	P4 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
}

void gradientDescentCube()
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	double sum0 = 0,sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	while(converge >= 0.01)
	//while(count --)
	{
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			h = hypothesisFuncion(TRAININGSET[i]);
			y = TRAININGSET[i].price;
			error[i] = h - y;
		}


		Jsum = 0;
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			Jsum += abs((error[i] * error[i] * error[i]))/(2*TRAININGSET.size());
		}

		newJ =  Jsum ;
		converge = oldJ - newJ;
		//cout << "Jsum : " << Jsum << endl;
		//cout << "oldJ : " << oldJ << endl;
		cout << "converge : " << converge << endl;
		oldJ = newJ;

		sum0 = 0;sum1 = 0;sum2 = 0;sum3 = 0;sum4 = 0;
		int sign;

		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			if(error[i] >= 0) sign = 1;
			else sign = -1;
			sum0 += (3  * sign * error[i] * error[i]);
			sum1 += (3  * sign * error[i] * error[i] * (TRAININGSET[i].sqft)) ;
			sum2 += (3  * sign * error[i] * error[i] * (TRAININGSET[i].floors));
			sum3 += (3  * sign * error[i] * error[i] * (TRAININGSET[i].bedrooms));
			sum4 += (3  * sign * error[i] * error[i] * (TRAININGSET[i].bathrooms));
		}
		
		P0 = P0 - (alphacube * sum0)/ (2 *TRAININGSET.size());
		P1 = P1 - (alphacube * sum1)/ (2 *TRAININGSET.size());
		P2 = P2 - (alphacube * sum2)/ (2 *TRAININGSET.size());
		P3 = P3 - (alphacube * sum3)/ (2 *TRAININGSET.size());
		P4 = P4 - (alphacube * sum4)/ (2 *TRAININGSET.size());

		count ++;
	}
	cout << count << endl;
}

void gradientDescentAbs()
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	double sum0 = 0,sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	while(converge >= 0.0001)
	//while(count --)
	{
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			h = hypothesisFuncion(TRAININGSET[i]);
			y = TRAININGSET[i].price;
			error[i] = h - y;
		}


		Jsum = 0;
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			Jsum += (abs(error[i]))/(2*TRAININGSET.size());
		}

		newJ =  Jsum ;
		converge = oldJ - newJ;
		//cout << "Jsum : " << Jsum << endl;
		//cout << "oldJ : " << oldJ << endl;
		cout << "converge : " << converge << endl;
		oldJ = newJ;

		sum0 = 0;sum1 = 0;sum2 = 0;sum3 = 0;sum4 = 0;
		int sign;

		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			if(error[i] >= 0)sign = 1;
			else sign = -1;

			sum0 += sign;
			sum1 += sign * (TRAININGSET[i].sqft)/(2 * TRAININGSET.size());
			sum2 += sign * (TRAININGSET[i].floors)/(2 * TRAININGSET.size());
			sum3 += sign * (TRAININGSET[i].bedrooms)/(2 * TRAININGSET.size());
			sum4 += sign * (TRAININGSET[i].bathrooms)/(2 * TRAININGSET.size());
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

void gradientDescent()
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	double sum0 = 0,sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	while(converge >= 0.01)
	//while(count--)
	{
		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			h = hypothesisFuncion(TRAININGSET[i]);
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
		cout << "converge : " << converge << endl;
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

double errorCalc()
{
	double tot = 0;
	for(int i = 0; i < TESTSET.size(); i++)
	{
		tot += (TESTSET[i].price - hypothesisFuncion(TESTSET[i])) * (TESTSET[i].price - hypothesisFuncion(TESTSET[i]));
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

	initialize();
	

	gradientDescentAbs();

	// for(i = 0; i < TRAININGSET.size(); i++)
	// {
	// 	cout << TRAININGSET[i].sqft << " " << TRAININGSET[i].floors << " " << TRAININGSET[i].bedrooms<< " " << TRAININGSET[i].bathrooms << " " << TRAININGSET[i].price << endl;
	// }

	cout << P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;

	for(int j = 0; j < TESTSET.size(); j++)
	{
		cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j]) << endl; 
	}
	cout << endl;
	cout << errorCalc() << endl;


	return 0;
}
