package com.mkis.assignments.logisticregression;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*Suppose that you are the administrator of a university department and
you want to determine each applicant's chance of admission based on their
results on two exams. You have historical data from previous applicants
that you can use as a training set for logistic regression. For each training
example, you have the applicant's scores on two exams and the admissions
decision.

With linear decision boundary.

Initialize thetas so gradient descent will not get stuck in a local optima!*/

public class LogisticRegression extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\logisticregression\\data1.txt";
    private static double m; // number of training examples
    private static int n; // number of features
    private static int iterations = 0;// number of iterations needed for gradient descent
    private static double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example

    //For plotting
    private LogisticRegression(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph for admitted students
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx1 = new XYSeries("Admitted");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == 1) seriesEx1.add(dataSet.get(i).xVariables[1], dataSet.get(i).xVariables[2]);
        }
        XYDataset dataSetVisEx1 = getData(seriesEx1);
        XYItemRenderer rendererDataEx1 = new XYLineAndShapeRenderer(false, true);
        rendererDataEx1.setBasePaint(Color.GREEN);
        ValueAxis xAxisDataEx1 = new NumberAxis("Exam 1 score");
        ValueAxis yAxisDataEx1 = new NumberAxis("Exam 2 score");

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSetVisEx1);
        plot.setRenderer(0, rendererDataEx1);
        plot.setDomainAxis(0, xAxisDataEx1);
        plot.setRangeAxis(0, yAxisDataEx1);

        //SETUP SCATTER graph for students not admitted
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx2 = new XYSeries("Not admitted");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == 0) seriesEx2.add(dataSet.get(i).xVariables[1], dataSet.get(i).xVariables[2]);
        }
        XYDataset dataSetVisEx2 = getData(seriesEx2);
        XYItemRenderer rendererDataEx2 = new XYLineAndShapeRenderer(false, true);
        rendererDataEx2.setBasePaint(Color.ORANGE);
        ValueAxis xAxisDataEx2 = new NumberAxis();
        ValueAxis yAxisDataEx2 = new NumberAxis();
        xAxisDataEx2.setAxisLineVisible(false);
        xAxisDataEx2.setVerticalTickLabels(false);
        yAxisDataEx2.setAxisLineVisible(false);
        yAxisDataEx2.setVerticalTickLabels(false);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(1, dataSetVisEx2);
        plot.setRenderer(1, rendererDataEx2);
        plot.setDomainAxis(1, xAxisDataEx2);
        plot.setRangeAxis(1, yAxisDataEx2);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Decision boundary");
        double x = 25;
        while (x <= 100) {
            lineSeries.add(x, (theta[0] + theta[1] * x) / (-1 * theta[2]));
            x += 1;
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only
        ValueAxis xAxisLine = new NumberAxis();
        ValueAxis yAxisLine = new NumberAxis();
        xAxisLine.setAxisLineVisible(false);
        xAxisLine.setVerticalTickLabels(false);
        yAxisLine.setAxisLineVisible(false);
        yAxisLine.setVerticalTickLabels(false);

        // Set the line data, renderer, and axis into plot
        plot.setDataset(2, lineDataSet);
        plot.setRenderer(2, rendererLine);
        plot.setDomainAxis(2, xAxisLine);
        plot.setRangeAxis(2, yAxisLine);

        // Map the line to the second xAxis and second yAxis
        /*plot.mapDatasetToDomainAxis(1, 0);
        plot.mapDatasetToDomainAxis(1, 0);
        plot.mapDatasetToRangeAxis(2, 0);
        plot.mapDatasetToRangeAxis(2, 0);*/

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - Logistic Regression", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {

        loadData();
        initTheta();
        System.out.println("Cost Function at theta: " + Arrays.toString(theta) + " : " + createCostFunction(dataSet));
        createGradients(dataSet);
        doGradientDescent(dataSet);
        System.out.println("Cost Function at theta: " + Arrays.toString(theta) + " : " + createCostFunction(dataSet));
        System.out.println("Accuracy of the model: " + calcAccuracyOfModel(dataSet) + " %");
        //doLBFGS(dataSet);

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("For a student with scores 45 and 85, we predict an admission probability: ");
        System.out.println(nf.format(100 * (createHypothesis(new double[]{1, 45, 85}))) + " %");
        System.out.println(predict(new double[]{1, 45, 85}) == 1 ? "He will be admitted." : "He will not be admitted.");

        LogisticRegression visualizationOfData = new LogisticRegression("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    //Get XYDataset data data for visualization
    private XYDataset getData(XYSeries series) {
        XYSeriesCollection xySeriesCollectionData = new XYSeriesCollection();
        xySeriesCollectionData.addSeries(series);
        return xySeriesCollectionData;
    }

    //Get XYDataset line data for visualization
    private XYDataset getLineData(XYSeries lineSeries) {
        XYSeriesCollection xySeriesCollectionLine = new XYSeriesCollection();
        xySeriesCollectionLine.addSeries(lineSeries);
        return xySeriesCollectionLine;
    }

    public static class Instance {
        double yValue;
        double[] xVariables;

        Instance(double yValue, double[] xVariables) {
            this.yValue = yValue;
            this.xVariables = xVariables;
        }
    }

    //Load the data from the txt file into an array
    private static void loadData() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            String[] columns;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                n = columns.length;
                double y = Double.parseDouble(columns[n - 1]);
                double xArray[] = new double[columns.length];
                xArray[0] = 1.0;
                for (int i = 0; i < columns.length - 1; i++) {
                    xArray[i + 1] = Double.parseDouble(columns[i]);
                }
                Instance instance = new Instance(y, xArray);
                dataSet.add(instance);
            }
            m = dataSet.size();
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Initialize theta
    private static void initTheta() {
        theta = new double[n];
        theta[0] = -20;
        theta[1] = 5;
        theta[2] = 1;
        /*for (int i = 0; i < n; i++) {
            theta[i] = 0.0;
        }*/
    }

    /*h (x) = 1/(1+exp(-z))  - Sigmoid function for h (x)
        z(1) = theta0*x0(1) + theta1*x1(1) + theta2 * x2(1)
        Cost = -y(1)*log(z(1))-(1-y(1))*log(1-z(1))
        m = 1 in above example (first tr. example values)
         */
    private static double createCostFunction(List<Instance> instances) {
        double cost = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            cost += -y * Math.log(createHypothesis(x)) - (1 - y) * Math.log(1 - createHypothesis(x));
        }
        return (1 / m) * cost;
    }

    //Create gradients
    private static void createGradients(List<Instance> instances) {
        double[] sum = new double[theta.length];
        for(int i=0; i< theta.length; i++){
            sum[i] = 0.0;
        }
        double[] grad = new double[theta.length];
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            for(int i=0; i< theta.length; i++){
                sum[i] += hypothesis * x[i] - y*x[i];
            }
        }
        for(int i=0; i< theta.length; i++){
            grad[i] = (1 / m) * sum[i];
        }
        System.out.println("Gradients with theta " + Arrays.toString(theta) + " : " + Arrays.toString(grad) + "");
    }

    //Sigmoid function
    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    //Creating our hypothesis (h(x)) / Prediction
    // h(x) = P(y=1 | x; theta)  <- Probability that y = 1, given x parameterized by theta.
    private static double createHypothesis(double[] x) {
        double hypothesis = 0.0;
        for (int i = 0; i < theta.length; i++) {
            hypothesis += theta[i] * x[i];
        }
        return sigmoid(hypothesis);
    }

    //Do gradient descent
    private static void doGradientDescent(List<Instance> instances) {
        for (int k = 0; k < 5000; k++) {
            double[] temp = new double[n];
            for (int i = 0; i < n; i++) {
                temp[i] = 0.0;
            }
            double costFunctionOld = createCostFunction(dataSet);
            for (Instance instance : instances) {
                double[] x = instance.xVariables;
                double hypothesis = createHypothesis(x);
                double y = instance.yValue;
                for (int i = 0; i < n; i++) {
                    temp[i] = temp[i] + (hypothesis-y)*x[i];
                }
            }
            double alpha = 0.00001;
            for (int i = 0; i < n; i++) {
                theta[i] = theta[i] - alpha * temp[i];
            }
            iterations++;
            //Cost function to descend, theta after each iteration:
            //System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta) + ", Cost function: " + createCostFunction(dataSet));
            if (costFunctionOld - createCostFunction(dataSet) < 0.0001) {
                System.out.println("Number of iterations: " + iterations);
                return;
            }
        }
        System.out.println("Number of iterations: " + iterations);
    }

    //Predict whether the label is 0 or 1 (not admitted or admitted) using
    //a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    private static double predict(double[] scores) {
        if(createHypothesis(scores) >= 0.5) return 1;
        return 0;
    }

    //Calculate accuracy of logistic regression
    private static double calcAccuracyOfModel (List<Instance> instances) {
        int counter = 0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            if (predict(x) != y) counter++;
        }
        return (1 - counter / m)*100;
    }

}
