/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MP1;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;
import weka.core.matrix.Matrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 *
 * @author Jessa
 */
public class MultivariateLR2 {

    double alpha = 1;
    int numIterations = 100;
    int m;

    public FeatureNormalizationValues1 featureNormalization(Matrix x) {
        FeatureNormalizationValues1 FNV = new FeatureNormalizationValues1();
        int r = x.getRowDimension();
        int c = x.getColumnDimension();
        Matrix Ave = new Matrix(1, 2);
        Matrix STD = new Matrix(1, 2);
        Matrix X_Norm = new Matrix(r, c);

        double sum = 0;
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < r; j++) {
                sum += x.get(j, i);
            }
            Ave.set(0, i, sum / r);
            sum = 0;
        }
        for (int i = 0; i < c; i++) {
            sum = 0;
            for (int j = 0; j < r; j++) {
                sum += Math.pow((x.get(j, i) - Ave.get(0, i)), 2);
            }
            sum /= r;
            STD.set(0, i, Math.sqrt(sum));
        }

        for (int i = 0; i < c; i++) {
            for (int j = 0; j < r; j++) {
                X_Norm.set(j, i, (x.get(j, i) - Ave.get(0, i)) / STD.get(0, i));
            }
        }

        FNV.setMu(Ave);
        FNV.setSigma(STD);
        FNV.setX(X_Norm);

        return FNV;
    }

    public double[][] load(String file) throws FileNotFoundException {
        double[][] a = null;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(file)));
            String st = "";
            a = new double[47][3];
            int i = 0, j = 0;
            while ((st = reader.readLine()) != null) {
                String[] s = st.split(",");
                for (String str : s) {
                    a[i][j++] = new Double(str).doubleValue();
                }
                j = 0;
                i++;
            }
            reader.close();
        } catch (Exception e) {
            System.out.println("No Such File");
        }
        return a;

    }

    public GradientDescentValues1 gradientDescentMulti1(Matrix x, Matrix y, Matrix theta) {
        GradientDescentValues1 GDV = new GradientDescentValues1();
        double[][] jHist = new double[numIterations][1];
        int n = y.getRowDimension();
        for (int i = 0; i < numIterations; i++) {
            jHist[i][0] = computeCostMulti(x, y, theta);
            System.out.println("Computed Cost: " + jHist[i][0]);
            Matrix a = (x.times(theta).minus(y)).transpose().times(x);
            theta.minusEquals(a.times(alpha / n).transpose());
            System.out.print("Theta Values:");
            theta.print(0, 3);
        }
        GDV.setTheta(theta);
        Matrix J_Hist = new Matrix(jHist);
        GDV.setCostHistory(J_Hist);
        plotGradient(J_Hist);
        return GDV;
    }

    public void plotGradient(Matrix J) {
        GradientDescentValues1 GDV = new GradientDescentValues1();
        double[] yData = new double[numIterations];
        // J.print(1, 2);
        double[][] f = J.getArray();
        double[] xData = new double[numIterations];

        for (int i = 0; i < numIterations; i++) {
            yData[i] = f[i][0];
            xData[i] = i;
        }
        Chart chart = QuickChart.getChart("Convergence Graph", "Number of Iterations", "CostJ", "Gradient Descent", xData, yData);
        new SwingWrapper(chart).displayChart();
    }

    public Matrix normalEquation(Matrix x, Matrix y) {
        Matrix x2 = x.transpose();
        Matrix x3 = x2.times(x).inverse().times(x2).times(y);
        return x3;
    }

    public Double computeCostMulti(Matrix x, Matrix y, Matrix theta) {
        Matrix d = x.times(theta).minus(y);
        Matrix sum = d.transpose().times(d);
        m = y.getRowDimension();
        double finValue = 1.0 / (2 * m) * sum.get(0, 0);
        return finValue;
    }
}

class FeatureNormalizationValues1 {

    Matrix X;
    Matrix mu;
    Matrix sigma;

    public Matrix getX() {
        return X;
    }

    public void setX(Matrix X) {
        this.X = X;
    }

    public Matrix getMu() {
        return mu;
    }

    public void setMu(Matrix mu) {
        this.mu = mu;
    }

    public Matrix getSigma() {
        return sigma;
    }

    public void setSigma(Matrix sigma) {
        this.sigma = sigma;
    }

}

class GradientDescentValues1 {

    Matrix theta;
    Matrix costHistory;

    public Matrix getTheta() {
        return theta;
    }

    public void setTheta(Matrix theta) {
        this.theta = theta;
    }

    public Matrix getCostHistory() {
        return costHistory;
    }

    public void setCostHistory(Matrix costHistory) {
        this.costHistory = costHistory;
    }
}

class TestMLR2 {

    public static void main(String[] args) throws FileNotFoundException {
        MultivariateLR2 MLR = new MultivariateLR2();
        GradientDescentValues1 GDV = new GradientDescentValues1();
        FeatureNormalizationValues1 FNV = new FeatureNormalizationValues1();

        long start = System.currentTimeMillis();
        System.out.println("Loading data...");

        double[][] M = MLR.load("ex1data2.txt");
        Matrix data = new Matrix(M);

        int m = data.getRowDimension();

        Matrix x = data.getMatrix(0, m - 1, 0, 1);//gets the features
        Matrix y = data.getMatrix(0, m - 1, 2, 2);//gets the output
        System.out.println("First 10 examples from the dataset:");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print("    " + M[i][j]);
            }
            System.out.println("    ");
        }
        System.out.print("Normalizing features...");
        FNV = MLR.featureNormalization(x);
        Matrix X = new Matrix(m, 3);
        //adds intercept
        X.setMatrix(0, m - 1, 0, 0, new Matrix(m, 1, 1.0));
        X.setMatrix(0, m - 1, 1, 2, FNV.X);
        Matrix theta = new Matrix(3, 1);//automatically assigns vector of zeros to the matrix

        System.out.println("Running Gradient Descent...");
        GDV = MLR.gradientDescentMulti1(X, y, theta);
        System.out.println("Theta from Gradient Descent:");
        theta.print(5, 3);
        System.out.println("Predicted price of a 1650 sq-ft 3br house...\n");

        Matrix inp = new Matrix(3, 1);
        //prediction code for GD
        inp.set(0, 0, 1);
        inp.set(1, 0, (1650 - FNV.mu.get(0, 0)) / FNV.sigma.get(0, 0));
        inp.set(2, 0, (3 - FNV.mu.get(0, 1)) / FNV.sigma.get(0, 1));
      
        double price = GDV.getTheta().transpose().times(inp).get(0, 0);
        System.out.println("GD Estimated Price " + price);
        System.out.println("Solving with normal equations...\n");
        Matrix theta2 = MLR.normalEquation(X, y);
        System.out.print("NE Theta Values:");
        theta2.print(2, 2);
        price = theta2.transpose().times(inp).get(0, 0);
        System.out.println("NE Estimated Price: " + price);
        Matrix x_NormEq = new Matrix(m, 3);

        long end = System.currentTimeMillis();
        long dif = end - start;
        if (dif > 1000) {
            dif = (end - start) / 1000;
            System.out.println("Speed:" + dif + " seconds");
        } else {
            System.out.println("Speed:" + dif + " milliseconds");
        }
    }
}
