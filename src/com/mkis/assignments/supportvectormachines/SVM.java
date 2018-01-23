package com.mkis.assignments.supportvectormachines;

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
import java.util.*;
import java.util.List;

/*In the first half of this exercise, you will be using support vector machines
(SVMs) with various example 2D datasets.*/

public class SVM extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\supportvectormachines\\data1.txt";
    private static double m; // number of training examples
    private static int n; // number of columns in dataset
    private static double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example
    private static List<Double> meansOfFeatures = new ArrayList<>(); //list containing the means of each feature
    private static List<Double> maxMinusMinOfFeatures = new ArrayList<>(); //list containing max-min of each feature

    //For plotting
    private SVM(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph for admitted students
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx1 = new XYSeries("1");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == 1)
                seriesEx1.add(dataSet.get(i).xVariables[1] * maxMinusMinOfFeatures.get(0) + meansOfFeatures.get(0),
                        dataSet.get(i).xVariables[2] * maxMinusMinOfFeatures.get(1) + meansOfFeatures.get(1)); //feature scaling
        }
        XYDataset dataSetVisEx1 = getData(seriesEx1);
        XYItemRenderer rendererDataEx1 = new XYLineAndShapeRenderer(false, true);
        rendererDataEx1.setBasePaint(Color.GREEN);
        ValueAxis xAxisDataEx1 = new NumberAxis("Exam 1 score");
        ValueAxis yAxisDataEx1 = new NumberAxis("Exam 2 score");
        xAxisDataEx1.setLowerBound(0);
        xAxisDataEx1.setUpperBound(5);
        yAxisDataEx1.setLowerBound(1.2);
        yAxisDataEx1.setUpperBound(4.8);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSetVisEx1);
        plot.setRenderer(0, rendererDataEx1);
        plot.setDomainAxis(0, xAxisDataEx1);
        plot.setRangeAxis(0, yAxisDataEx1);

        //SETUP SCATTER graph for students not admitted
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx2 = new XYSeries("2");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == -1)
                seriesEx2.add(dataSet.get(i).xVariables[1] * maxMinusMinOfFeatures.get(0) + meansOfFeatures.get(0),
                        dataSet.get(i).xVariables[2] * maxMinusMinOfFeatures.get(1) + meansOfFeatures.get(1)); //feature scaling
        }
        XYDataset dataSetVisEx2 = getData(seriesEx2);
        XYItemRenderer rendererDataEx2 = new XYLineAndShapeRenderer(false, true);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(1, dataSetVisEx2);
        plot.setRenderer(1, rendererDataEx2);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Hyperplane");
        double x = 0.0;
        while (x <= 5) {
            lineSeries.add(x,
                    ((theta[0] + theta[1] * ((x - meansOfFeatures.get(0)) / maxMinusMinOfFeatures.get(0))) / (-1 * theta[2])) * maxMinusMinOfFeatures.get(1) + meansOfFeatures.get(1)); //feature scaling
            x += 0.05;
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only

        // Set the line data, renderer, and axis into plot
        plot.setDataset(2, lineDataSet);
        plot.setRenderer(2, rendererLine);

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - SVM", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {

        getNumberOfFeaturesAndTrainingExamples();
        loadData();
        init();
        System.out.println("Hinge loss at theta: " + Arrays.toString(theta) + " : " + createLossFunction(dataSet));
        doGradientDescent(dataSet);
        System.out.println("Hinge loss at theta: " + Arrays.toString(theta) + " : " + createLossFunction(dataSet));

        SVM visualizationOfData = new SVM("Visualization of data");
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

    private static void getNumberOfFeaturesAndTrainingExamples() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            String[] columns;
            List<List<Double>> tempList = new ArrayList<>();
            m = 0;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                n = columns.length;
                if (m == 0) {
                    for (int i = 0; i < n - 1; i++) {
                        List<Double> tempInnerList = new ArrayList<>();  //create inner lists, number depends on the number of features
                        tempList.add(i, tempInnerList);
                    }
                }
                for (int i = 0; i < n - 1; i++) {
                    tempList.get(i).add(Double.parseDouble(columns[i]));
                }
                m++;
            }
            //for mean normalization get means and (max-min)s of all the features
            for (int i = 0; i < n - 1; i++) {
                meansOfFeatures.add(i, tempList.get(i).stream().mapToDouble(val -> val).average().getAsDouble());
                maxMinusMinOfFeatures.add(i, Collections.max(tempList.get(i)) - Collections.min(tempList.get(i)));
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
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
                double y;
                // y  in {-1,1}
                if (Double.parseDouble(columns[n - 1]) == 0) {
                    y = -1d;
                } else {
                    y = 1d;
                }
                double xArray[] = new double[n];
                xArray[0] = 1;
                for (int i = 0; i < n-1; i++) {
                    xArray[i+1] = (Double.parseDouble(columns[i]) - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i); // feature scaling applied, to get variables between 0 and 1
                }
                //System.out.println(xArray[0] + " | " + xArray[1] + " | " + xArray[2] + " | " + y);
                Instance instance = new Instance(y, xArray);
                dataSet.add(instance);
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Random initialize theta
    private static void init() {
        theta = new double[n];
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            theta[i] = (double) random.nextInt(100) / 100;
        }
    }

    //Hinge loss
    private static double createLossFunction(List<Instance> instances) {
        double cost = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            if (y * createHypothesis(x) >= 1) {
                cost += 0;
            } else {
                cost += - y * createHypothesis(x);
            }
        }
        //System.out.println("Cost function value with theta " + Arrays.toString(theta) + ": " + J);
        return cost;
    }

    //Creating our hypothesis (h(x))
    private static double createHypothesis(double[] x) {
        double hypothesis = 0.0;
        for (int i = 0; i < n; i++) {
            hypothesis += theta[i] * x[i];
        }
        return hypothesis;
    }

    //Do gradient descent
    private static void doGradientDescent(List<Instance> instances) {
        double alpha = 0.01; //learning rate
        double C = 100; //reg. term, higher - better fit the dataset, lower - more general
        for (int it = 1; it < 10000; it++) {
            double[] temp = new double[n];
            for (int i = 0; i < n; i++) {
                temp[i] = 0.0;
            }
            for (Instance instance : instances) {
                double[] x = instance.xVariables;
                double hypothesis = createHypothesis(x);
                double y = instance.yValue;
                if (y * hypothesis < 1) {
                    for (int i = 0; i < n; i++) {
                        temp[i] +=  x[i] * y;
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                theta[i] = (1 - alpha) * theta[i] + C * temp[i];
            }
            //Stop when |hinge loss| < 0.1
            if (Math.abs(createLossFunction(dataSet)) < 0.1) return;
            //System.out.println("Iteration: " + it + " Loss: " + createLossFunction(dataSet));
        }
    }

}
