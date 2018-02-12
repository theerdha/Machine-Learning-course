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
#define N 5

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

// Function to get cofactor of A[p][q] in temp[][]. n is current dimension of A[][]
void getCofactor(double A[N][N], double temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
/* Recursive function for finding determinant of matrix. n is current dimension of A[][]. */
double determinant(double A[N][N], int n)
{
    double D = 0; // Initialize result
    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];
 	double temp[N][N]; // To store cofactors
    int sign = 1;  // To store sign multiplier
	// Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}
 
// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(double A[N][N],double adj[N][N])
{
    if (N == 1)
    {
        adj[0][0] = 1;
        return;
    }
    // temp is used to store cofactors of A[][]
    int sign = 1;
    double temp[N][N];
 
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, N);
            // sign of adj[j][i] positive if sum of row and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1; 
            // Interchanging rows and columns to get the transpose of the cofactor matrix
            adj[j][i] = (sign)*(determinant(temp, N-1));
        }
    }
}
 
// Function to calculate and store inverse, returns false if matrix is singular
bool inverse(double A[N][N], double inverse[N][N])
{
    // Find determinant of A[][]
    double det = determinant(A, N);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
    // Find adjoint
    double adj[N][N];
    adjoint(A, adj);
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i][j] = adj[i][j]/det;
 
    return true;
}

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

void gradientDescentReg()
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
		Jsum += (lambda * (P1 * P1 + P2 * P2 + P3 * P3 + P4 * P4)) / (2*TRAININGSET.size());

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
		P1 = P1 * (1 - (alpha * lambda/TRAININGSET.size())) - (alpha * sum1);
		P2 = P2 * (1 - (alpha * lambda/TRAININGSET.size())) - (alpha * sum2);
		P3 = P3 * (1 - (alpha * lambda/TRAININGSET.size())) - (alpha * sum3);
		P4 = P4 * (1 - (alpha * lambda/TRAININGSET.size())) - (alpha * sum4);

		count ++;
	}
	cout << count << endl;
}

void gradientDescentIRLS()
{
	double psi[TRAININGSET.size()][5];
	double psiT [5][TRAININGSET.size()];
	double prod [5][5];
	double prodI [5][5];
	double prodIpsiT[5][TRAININGSET.size()];

	double y[TRAININGSET.size()];
	for(int rows = 0; rows < TRAININGSET.size(); rows++)
	{
		for(int columns = 0; columns < 5; columns++)
		{
			if(columns == 0) psi[rows][columns] = 1;
			else if(columns == 1)psi[rows][columns] = TRAININGSET[rows].sqft;
			else if(columns == 2)psi[rows][columns] = TRAININGSET[rows].floors;
			else if(columns == 3)psi[rows][columns] = TRAININGSET[rows].bedrooms;
			else psi[rows][columns] = TRAININGSET[rows].bathrooms;
		}
	}

	for(int rows = 0; rows < TRAININGSET.size(); rows++)
	{
		y[rows] = TRAININGSET[rows].price;
	}

	for(int rows = 0; rows < 5; rows++)
	{
		for(int columns = 0; columns < TRAININGSET.size(); columns++)
		{
			psiT[rows][columns] = psi[columns][rows];
		}
	}

	for(int rows = 0; rows < 5; rows++)
	{
		for(int columns = 0; columns < 5; columns++)
		{
			prod[rows][columns] = 0;
		}
	}

	for(int rows = 0; rows < 5; rows++)
	{
		for(int columns = 0; columns < TRAININGSET.size(); columns++)
		{
			prodIpsiT[rows][columns] = 0;
		}
	}

	for(int i = 0; i < 5; ++i)
    {
    	for(int j = 0; j < 5; ++j)
        {   
        	for(int k = 0; k < TRAININGSET.size(); ++k)
            {
                prod[i][j] += psiT[i][k] * psi[k][j];
            }
        }
    } 

   bool b = inverse(prod,prodI);
   if(b == false) return;

   //prodI * psiT

   for(int i = 0; i < 5; ++i)
    {
    	for(int j = 0; j < TRAININGSET.size(); ++j)
        {   
        	for(int k = 0; k < 5; ++k)
            {
                prodIpsiT[i][j] += prodI[i][k] * psiT[k][j];
            }
        }
    } 

    for(int i = 0; i < 5; ++i)
    {
    	for(int j = 0; j < TRAININGSET.size(); ++j)
        {  
        	if(i == 0)P0 += prodIpsiT[i][j] * y[j];
        	if(i == 1)P1 += prodIpsiT[i][j] * y[j];
        	if(i == 2)P2 += prodIpsiT[i][j] * y[j];
        	if(i == 3)P3 += prodIpsiT[i][j] * y[j];
        	if(i == 4)P4 += prodIpsiT[i][j] * y[j];
        }
    }


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
	

	gradientDescentIRLS();

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
