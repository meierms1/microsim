//Matrix operations
#include <stdio.h>
#include <math.h>
// #include "functions1.h"
// #include "open.h"
void matinvnew(double **coeffmatrix,double **inv, long size);
void multiply(double **inv,double *y,double *prod, long size);
void multiply2d(double **m1,double **m2,double **prod, long size);
void vectorsum(double *y1,double *y2, double *sum, long size);
void substituteb(double **fac,double *y,double *vec, long size);
void substitutef(double **fac,double **y1,int index,double *vec, long size);
void pivot(double **coeffmatrix,double **factor,int k,int *tag, long size);
void colswap(double **m1,double **m2,int *tag, long size);
void rowswap(double **m1,double **m2,int *tag, long size);
//Sum of two vectors
void vectorsum(double *y1,double *y2, double *sum, long size) {
	int j;
	for(j=0;j < (size);j++)
               sum[j]=y1[j]+y2[j];
}
//Multipliction of matrix and vector
void multiply(double **inv,double *y,double *prod,long size) {
	int i,j,k;
	double sum;
	for(i=0;i<size;i++)
	{
		sum=0;
		for(j=0;j<size;j++)
			sum=sum+inv[i][j]*y[j];
		prod[i]=sum;
	}

}
//Multiplication of two matrices
void multiply2d(double **m1,double **m2,double **prod,long size) {
	int i,j,k;
	double sum;
	for(k=0;k<size;k++)
	{
		for(i=0;i<size;i++)
		{
			sum=0;
			for(j=0;j<size;j++)
				sum=sum+m1[k][j]*m2[j][i];
			prod[k][i]=sum;
		}
	}
}
//Matrix inversion using LU decomposition
void matinvnew(double **coeffmatrix,double **inv,long size) {
	int i,j,k,p,q,tag[size];
	double **factor,**iden, **inv1, **prod;
	double *vec1,*vec,fact;
	factor = MallocM(size,size);
	inv1   = MallocM(size,size);
	iden   = MallocM(size,size);
	prod   = MallocM(size,size);
	vec1   = MallocV(size);
	vec    = MallocV(size);
	//Making the Upper Triangular Matrix.
	for(k=0; k <size; k++)
		tag[k]=k;
	for(k=0; k < size; k++) {
		pivot(coeffmatrix,factor,k,tag,size);
		for(i=k+1; i < size;i++)
		{
			fact=-coeffmatrix[i][k]/coeffmatrix[k][k];
			factor[i][k]=-fact;
			for(j=k;j<=(size-1);j++)
				coeffmatrix[i][j]=fact*coeffmatrix[k][j]+coeffmatrix[i][j];
		}
	}
	for(i=0;i < size; i++) {
		for(j=0;j<size;j++)
		{
			if(i==j)
				factor[i][j]=1;
			if(j>i)
				factor[i][j]=0;
		}
	}
	/*************************************************************************/
	//The Identity Matrix.
	for(i=0;i<(size);i++)
	{
		for(j=0;j<(size);j++)
		{
			if(i==j)
				iden[i][j]=1;
			else
				iden[i][j]=0;
		}
	}
	/*************************************************************************/
	//Forward and backward substitution to get the final identity matrix. 
	for(i=0;i<(size);i++)
	{
		substitutef(factor,iden,i,vec1,size);
		substituteb(coeffmatrix,vec1,vec,size);
		for(j=0;j<(size);j++)
			inv1[j][i]=vec[j];
	}
	/**************************************************************************/
	colswap(inv1,inv,tag,size);
	multiply2d(factor,coeffmatrix,prod,size);
	rowswap(prod,coeffmatrix,tag,size);
	
	FreeM(factor,size);
	FreeM(iden,size);
	FreeM(inv1,size);
	FreeM(prod,size);
	free(vec1);
	free(vec);
}
/***************************************************************************************/
//Back Substitution.
void substituteb(double **fac,double *y,double *vec,long size)
{
	int i,j;
	double sum;
	vec[size-1]=y[size-1]*pow(fac[size-1][size-1],-1);
	for(i=(size-2);i>=0;i--)
	{
		sum=0;
		for(j=i+1;j<(size);j++)
			sum=sum-fac[i][j]*vec[j];
		vec[i]=(y[i]+sum)*pow(fac[i][i],-1);
	}
}
/*********************************************************************************************/
//Forward Substitution.
void substitutef(double **fac,double **y1,int index,double *vec,long size)
{
	int i,j;
	double d[size],sum;
	
	for(i=0;i<size;i++)
		d[i]=y1[i][index];
  	vec[0]=d[0];
	for(i=1;i<size;i++)
	{
		sum=0;
		for(j=0;j<i;j++)
			sum=sum-fac[i][j]*vec[j];
		vec[i]=d[i]+sum;
	}
// 	for(i=0;i<(size);i++)
// 		newmat[i][index]=vec[i];
}
/********************************************************************************************/
//Modulus operator
double mod(double k)
{
	if(k<0)
		return(-k);
	else
		return(k);
}
/********************************************************************************************/
//Pivoting.
void pivot(double **coeffmatrix,double **factor,int k,int *tag,long size)
{
	double swap,big;
	int tagswap,i,p,q,tag1;
	big=mod(coeffmatrix[k][k]);
	tag1=k;
	for(i=k+1;i<(size);i++)
	{
		if(mod(coeffmatrix[i][k])>big)
		{
			tag1=i;
			big=coeffmatrix[i][k];
		}
	}
	tagswap=tag[k];
	tag[k]=tag[tag1];
	tag[tag1]=tagswap;
	
	for(i=0;i<(size);i++)
	{
		swap=coeffmatrix[k][i];
		coeffmatrix[k][i]=coeffmatrix[tag1][i];
		coeffmatrix[tag1][i]=swap;
	}
	for(i=0;i<k;i++)
	{
		swap=factor[k][i];
		factor[k][i]=factor[tag1][i];
		factor[tag1][i]=swap;
	}
}
/*******************************************************************************************/			//Swapping Coloumns To get the final identity matrix because of the initial swappping for pivoting.
void colswap(double **m1,double **m2,int *tag, long size)
{	
	int i,j,k,p;
	for(k=0;k < size;k++)
	{
		for(j=0;j<size;j++)
		{
			for(p=0;p <size; p++)
				m2[p][tag[j]]=m1[p][j];
		}
	}
}
/********************************************************************************************/	 
//Switching rows
void rowswap(double **m1,double **m2,int *tag,long size)
{	
	int i,j,k,p;
	for(k=0;k<(size);k++)
	{
		for(j=0;j< size ;j++)
		{
			for(p=0; p < size;p++)
				m2[tag[j]][p]=m1[j][p];
		}
	}
}
/********************************************************************************************/