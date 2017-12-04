package com.mkis.assignments.gradientdescentandnormalequation;

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

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//First weekly assignments part 1 Linear Regression with gradient descent (non vectorized solution)

public class GradientDescentLinearRegressionOneVariable extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data1.txt";
    private static double m; // number of training examples
    private static int n; // number of features + 1
    private static double alpha = 0.01; // learning rate
    private static int iterations = 0; // number of iterations for gradient descent
    private static double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example

    private GradientDescentLinearRegressionOneVariable(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph
        // Create the scatter data, renderer, and axis
        XYSeries series = new XYSeries("Profits depending on city population");
        for (int i = 0; i < dataSet.size(); i++) {
            series.add(dataSet.get(i).xVariables[1], dataSet.get(i).yValue);
        }
        XYDataset dataSetVis = getData(series);
        XYItemRenderer rendererData = new XYLineAndShapeRenderer(false, true);   // Shapes only
        ValueAxis xAxisData = new NumberAxis("Population of City in 10,000s");
        ValueAxis yAxisData = new NumberAxis("Profit in $10,000s");
        xAxisData.setLowerBound(4);
        xAxisData.setUpperBound(25);
        yAxisData.setLowerBound(-5);
        yAxisData.setUpperBound(25);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSetVis);
        plot.setRenderer(0, rendererData);
        plot.setDomainAxis(0, xAxisData);
        plot.setRangeAxis(0, yAxisData);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Regression function");
        for (int i = 0; i < dataSet.size(); i++) {
            lineSeries.add((double) i, theta[0] + theta[1] * i);
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only
        ValueAxis xAxisLine = new NumberAxis();
        ValueAxis yAxisLine = new NumberAxis();
        xAxisLine.setLowerBound(4);
        xAxisLine.setUpperBound(25);
        xAxisLine.setAxisLineVisible(false);
        yAxisLine.setLowerBound(-5);
        yAxisLine.setUpperBound(25);
        yAxisLine.setAxisLineVisible(false);

        // Set the line data, renderer, and axis into plot
        plot.setDataset(1, lineDataSet);
        plot.setRenderer(1, rendererLine);
        plot.setDomainAxis(1, xAxisLine);
        plot.setRangeAxis(1, yAxisLine);

        // Map the line to the second xAxis and second yAxis
        plot.mapDatasetToDomainAxis(1, 1);
        plot.mapDatasetToRangeAxis(1, 1);

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - Linear Regression", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {

        loadData();
        initTheta();
        createCostFunction(dataSet);
        doGradientDescent(dataSet);

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("The value (profit) prediction in $s for city (population) size: 35,000:");
        System.out.println(nf.format((theta[0] + theta[1] * 3.5) * 10000));

        GradientDescentLinearRegressionOneVariable visualizationOfData =
                new GradientDescentLinearRegressionOneVariable("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    public static class Instance {
        double yValue;
        double[] xVariables;

        Instance(double yValue, double[] xVariables) {
            this.yValue = yValue;
            this.xVariables = xVariables;
        }
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

    //load the data from the txt file into an array
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

    //initiate theta
    private static void initTheta() {
        theta = new double[n];
        for (int i = 0; i < n; i++) {
            theta[i] = 0.0;
        }
    }

    private static double createHypothesis(double[] x) {
        double hypothesis = 0.0;
        for (int i = 0; i < theta.length; i++) {
            hypothesis += theta[i] * x[i];
        }
        return hypothesis;
    }

    //Create Cost function
    private static double createCostFunction(List<Instance> instances) {
        double sumOfsqrErrors = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            sumOfsqrErrors += Math.pow((hypothesis - y), 2);
        }
        double J = (1 / (2 * m)) * sumOfsqrErrors;
        //System.out.println("Cost function value with theta(" + Arrays.toString(theta) + "): " + J);
        return J;
    }

    private static boolean doGradientDescent(List<Instance> instances) {
        double temp0 = 0.0;
        double temp1 = 0.0;
        double costFunctionOld = createCostFunction(dataSet);
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            temp0 += (hypothesis - y) * x[0];
            temp1 += (hypothesis - y) * x[1];
        }
        theta[0] = theta[0] - (alpha / m) * temp0;
        theta[1] = theta[1] - (alpha / m) * temp1;
        iterations++;
         /*Cost function to descend, theta after each iteration:
        System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta));
        createCostFunction(dataSet);*/
        return (costFunctionOld - createCostFunction(dataSet)) < 0.0000001 || doGradientDescent(dataSet);
    }

}
