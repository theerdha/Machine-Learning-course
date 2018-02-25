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

void gradientDescentReg(double lambda)
{
	double h,y,newJ,oldJ = DBL_MAX,Jsum,converge = 1000;
	double sum0 = 0,sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
	vector <double> error(TRAININGSET.size());
	int count = 0;
	while(converge >= 0.01)
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
		Jsum += (lambda * (P1 * P1 + P2 * P2 + P3 * P3 + P4 * P4));

		newJ =  Jsum ;
		converge = oldJ - newJ;

		if(count == 100 || count == 200 || count == 500 || count == 1000 || count == 1500 || count == 2000 || count == 5000 || count == 10000)
		{
			cout << "error after " << count << " iterations : " << Jsum << "   ";
			cout << "convergence : " << converge << endl;
		}
		//cout << "Jsum : " << Jsum << endl;
		//cout << "oldJ : " << oldJ << endl;
		//cout << "converge : " << converge << endl;
		oldJ = newJ;

		sum0 = 0;sum1 = 0;sum2 = 0;sum3 = 0;sum4 = 0;

		for(int i = 0; i < TRAININGSET.size(); i++)
		{
			sum0 += error[i];
			sum1 += error[i] * (TRAININGSET[i].sqft);
			sum2 += error[i] * (TRAININGSET[i].floors);
			sum3 += error[i] * (TRAININGSET[i].bedrooms);
			sum4 += error[i] * (TRAININGSET[i].bathrooms);
		}
		
		P0 = P0 - (alpha * sum0/ TRAININGSET.size());
		//cout << "size : " << TRAININGSET.size() << endl;
		P1 = P1 * (1 - ((alpha * lambda)/TRAININGSET.size())) - (alpha * sum1/ TRAININGSET.size());
		P2 = P2 * (1 - ((alpha * lambda)/TRAININGSET.size())) - (alpha * sum2/ TRAININGSET.size());
		P3 = P3 * (1 - ((alpha * lambda)/TRAININGSET.size())) - (alpha * sum3/ TRAININGSET.size());
		P4 = P4 * (1 - ((alpha * lambda)/TRAININGSET.size())) - (alpha * sum4/ TRAININGSET.size());

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
		if(count == 100 || count == 200 || count == 500 || count == 1000 || count == 1500 || count == 2000 || count == 5000 || count == 10000)
		{
			cout << "error after " << count << " iterations : " << Jsum << "   ";
			cout << "convergence : " << converge << endl;
		}
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
	ofstream outfile;
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
	gradientDescent();
	// for(i = 0; i < TRAININGSET.size(); i++)
	// {
	// 	cout << TRAININGSET[i].sqft << " " << TRAININGSET[i].floors << " " << TRAININGSET[i].bedrooms<< " " << TRAININGSET[i].bathrooms << " " << TRAININGSET[i].price << endl;
	// }

	cout << "parameters in gradient descent" <<  P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;

	// for(int j = 0; j < TESTSET.size(); j++)
	// {
	// 	cout << "actual : " << TESTSET[j].price << " Prediceted : " << hypothesisFuncion(TESTSET[j]) << endl; 
	// }
	
	cout << "error in gradient descent is " << errorCalc() << endl;
	cout << endl;
	cout << endl;

	double lambda = 0;
	outfile.open("pyinput.txt", std::ios::app);
	for(int  i = 0; i < 30 ; i++)
	{
		outfile << lambda << " ";
		lambda += 0.1;
	}
	outfile << endl;

	lambda = 0;
	for(int  i = 0; i < 30 ; i++)
	{
		initialize();
		gradientDescentReg(lambda);
		cout << "parameters in gradient descent with reg param " << lambda << " is " <<  P0 << " " << P1 << " " << P2 << " " << P3 << " " << P4 << endl;
		cout << "error in gradient decent with reg param " <<  lambda << " is " << errorCalc() << endl;
		outfile << errorCalc() << " ";
		cout << endl;
		cout << endl;
		lambda += 0.1;
	}

	outfile << endl;
	outfile.close();
	return 0;
}
