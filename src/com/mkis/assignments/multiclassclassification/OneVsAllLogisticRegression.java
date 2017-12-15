package com.mkis.assignments.multiclassclassification;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

/*For this exercise, you will use logistic regression and neural networks to
recognize handwritten digits (from 0 to 9). Automated handwritten digit
recognition is widely used today - from recognizing zip codes (postal codes)
on mail envelopes to recognizing amounts written on bank checks. This
exercise will show you how the methods you've learned can be used for this
classification task.
There are 5000 training examples (originally .mat file), where each training
example is a 20 pixel by 20 pixel grayscale image of the digit.

The .mat file were converted to a txt file with Octave.
One instance/training example is 400 variables (greyscale pixel float),
and 1 (401st attribute) class variable (1-10, 10 is the zero!).
Then to do the one vs all method 10 separate txt files were created where the "one" is represented with the number 1, and "all" the others with the number 0*/

public class OneVsAllLogisticRegression {

    private static double m; // number of training examples
    private static int n; // number of features
    private static int iterations = 0;// number of iterations needed for gradient descent
    private static Double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example
    private static List<Double> meansOfFeatures = new ArrayList<>(); //list containing the means of each feature
    private static List<Double> maxMinusMinOfFeatures = new ArrayList<>(); //list containing max-min of each feature
    private static NumberFormat nf = new DecimalFormat("##.##");
    private static Map<Integer, Double> probabilities = new HashMap();

    public static void main(String[] args) throws java.lang.Exception {

        //createTXTsForOneVsAll();

        //Result for this number should be 1
        double[] testValue1 = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.957244008714628e-005, 9.681627328690711e-006, -0.006384293300653621, -0.01394318491285407, -0.01404956427015255, -0.005171568627450985, 0.0004045901104724694, 1.702069716775609e-005, 0, 0, 0, 0, 0, 0, 0, 0, 0.0003026591751062882, -0.002368024132730029, -0.01004891769597661, -0.01134401369695497, -0.01099679996738829, -0.02720469691241395, 0.01141917369858516, 0.07907836003424205, 0.08168847580612257, 0.02990950226244327, -0.002359948305894271, -9.681627328686139e-005, 0, 0, 0, 0, 0, 0, 0, 0.0003952205882352858, -0.004576963841669685, -0.005383612472766361, 0.05147446895424914, 0.06329595588235364, 0.05327563316993536, 0.2120090395825695, 0.5897238391884543, 0.8285660573257086, 0.8168014025054476, 0.2814372957516341, -0.05806693571399416, -0.0348932802287575, -0.03310569852941109, -0.0173893995098034, -0.01250592320261392, -0.001198483551424667, 0.0002124183006535867, 0, 0, -0.006323018790849694, 0.02442883494354061, 0.468292211328976, 0.7904781284041396, 0.8074523080065361, 0.7986334082244011, 0.9084144381938497, 1.024130718954248, 1.019872889433551, 0.9985904479847496, 0.6986989379084966, 0.5243639000991943, 0.5405682359749457, 0.5277518892973858, 0.242023386437909, 0.1544472869008718, 0.01200552362317065, -0.002410471132897618, 0, 0, -0.01381399782135081, 0.07960662019485519, 0.8094158666939001, 0.9941447780501093, 0.9797096609477127, 0.9791355187908501, 0.9727073227073229, 1.003400241693901, 1.031912207244009, 1.038768654684096, 1.047597545615469, 1.054820533202887, 1.038917058142702, 1.027618447031591, 0.9995321520969512, 0.9674702478213516, 0.127650320003261, -0.01908905228758178, 0, 0, -0.002939644607843136, 0.01292736741266137, 0.2066798236655767, 0.2947748502178642, 0.2868996119281039, 0.2843406181917205, 0.3019004864593092, 0.43493683619281, 0.5225281692538118, 0.5422468341503259, 0.5384303002450972, 0.5475052314758185, 0.7280313010620907, 0.8394318321078427, 0.8395487472766878, 0.771504578567537, 0.09382684154742937, -0.01458792892156865, 0, 0, 0.0003791121438180305, -0.001770800254351682, -0.02576331988096716, -0.03590151237210092, -0.03502507031918826, -0.03525335290041203, -0.03370337722789754, -0.02198244058538213, -0.0141025131466312, -0.01230341200929477, -0.01327402062696222, -0.01049498148678822, 0.04831590640414128, 0.08455270881741436, 0.08385206473441734, 0.05604184501243283, 0.002853447515303855, -0.0007454853043088279, 0, 0, 0, 0, 0, 0, 0, 4.084967320261461e-005, -0.0002364355305531773, -0.002335409858387809, -0.003730766612200448, -0.004046160130718967, -0.003924972766884543, -0.004224416724416732, -0.01056134259259262, -0.01446589052287586, -0.01438929738562096, -0.01151790577342051, -0.001079756226815049, 0.0001923338779956439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // sample values
        //Result for this number should be 8
        double[] testValue2 = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0003465073529411829, -0.00473958333333333, -0.02151225490196081, -0.01160906862745099, 0.0002247156658921452, 8.88480392156865e-005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001121030532795374, -0.002197440087146109, 0.008064916938997437, 0.1615573767701532, 0.06955385348583901, -0.03881169064992651, -0.007773948120915474, 0.00069146582244005, -8.351401753638584e-017, -3.979127152443018e-016, -5.417280696623531e-016, -4.406808523758124e-016, -3.21004736348258e-016, -1.289577558754463e-017, 2.193370757350635e-018, 0, 0, 0, 0, 0, -0.0008458684929273157, -0.01783243123638382, 0.4340163739106746, 0.5940524748093678, 0.5220493259803919, 0.4169510007745297, 0.06814448869825653, -0.02218996800108935, -0.01682155501089334, -0.009988085511982785, -0.002059978530567035, -0.002581699346405454, -0.01089358660130738, -0.01210784313725495, -0.001935253267973856, 0.0002690473278708604, 0, 0, 0, 0, -0.0162524722632343, 0.1255806088894321, 0.6078305001834414, 0.104762409468291, 0.133183471933472, 0.4872983914197077, 0.5985057851969617, 0.1320617789735434, 0.2203982036334976, 0.5400912960471788, 0.6279279533529261, 0.5192449859361625, 0.4611475106328048, 0.1013678101178102, -0.0173130426806899, -0.006921701430690109, 0.0005584770290652706, 0, 0, 0, -0.02637356813827428, 0.2319595928649238, 0.4830052764161219, 0.005094754221132592, -0.01894109136710225, -0.01771450749391973, 0.31623503880719, 0.7535735634531593, 0.7315601511437912, 0.2817723651960778, 0.05748155395214164, 0.07212438725490172, 0.3163511029411762, 0.5868279888344223, 0.3470757761437908, 0.01651933607815979, -0.004664351851851876, 0, 0, 0, -0.0442374954139664, 0.412064133986928, 0.3913019812091504, -0.02220178036492385, -0.0004824857026143189, -0.02994399942929378, 0.1916504459422668, 0.9265284245642699, 0.59571502246732, -0.06183653322440053, -0.008995794436970891, -0.007173917483660046, -0.02371880106209138, 0.1174603077342045, 0.5260899714052287, 0.3066887645564119, -0.0144292960239651, 0, 0, 0, -0.04167828461946148, 0.3819611077069716, 0.4826250510620917, -0.01090737336601348, -0.005803496051198121, -0.005138549182667313, 0.500943848720044, 0.5132134565631803, 0.7002200095315914, 0.3213481243191719, -0.0248791664968144, -0.001546432461873811, -0.0005340073529411849, -0.03440592660675374, 0.1568772977941177, 0.563339634883753, -0.005919968681916994, 0, 0, 0, -0.0183351677469326, 0.143041445397603, 0.6654406658496731, 0.06590199482570777, -0.02577762459150298, 0.1023723723723724, 0.607584831154684, 0.0801732877178649, 0.155405739379085, 0.7353182870370367, 0.1970952570217279, -0.0342126225490196, -0.009583605664488039, -0.0007970281862745303, -0.01255060253267929, 0.6367634149987095, 0.0003545411220046373, 0, 0, 0, -0.0008066843336502097, -0.0178592902857613, 0.4476291563056268, 0.412750465397524, -0.02949118802059978, 0.2655533060082405, 0.6334021238432999, -0.001610037639449387, -0.02926976750506237, 0.2683994231788353, 0.7086527115633152, 0.3331662318427028, 0.05637575924340599, -0.06067847825200802, 0.2742502343972937, 0.6016978983103484, -0.009577167665402763, 0, 0, 0, 0.0002405120052178905, -0.009118395969499106, 0.07533626089324628, 0.6611453397331157, 0.5165614106753803, 0.7412798941475407, 0.5192070908224393, -0.03874446827342035, -0.001874659586056774, -0.02599026416122001, 0.1980993097169571, 0.6307945431644881, 0.609175994008714, 0.4690373944716766, 0.7273390182461872, 0.3298632342749994, -0.01629136029411761, 0, 0, 0, 0, -0.0004142156862745307, -0.004176981209150316, 0.1271875680827882, 0.2498647535403042, 0.7078050222167869, 0.2311962145969493, -0.0135269437636165, -0.0007306985294117535, 0.0007223583877995586, -0.02388003954180424, 0.0172916156045752, 0.2392310900054465, 0.4825332414215681, 0.2059516441993458, -0.009975643063878454, -0.002222052015250556, 0, 0, 0, 0, 0.0001084558823529449, -0.0003725490196078417, -0.0186614583333334, -0.03041952614379086, 0.04657428355957795, -0.00767003676470584, -8.986928104575732e-005, 8.272058823529417e-005, 2.757352941176478e-005, 0.0005952672129142791, -0.007719771241830043, -0.02393583197167753, -0.01855662785947702, -0.02486264297385618, -0.002455056866821562, 0.0003676470588235302, 0, 0, 0, 0, 0, 0, 4.900083606847242e-018, 3.063725490196124e-005, -0.005789867922220632, -0.0008731617647058198, 7.659313725489651e-005, 0, 0, 0, 1.776960784313526e-005, -0.0002375408496731871, -0.00146890318627441, -9.865196078430612e-005, 1.834413599119373e-005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        //Randomly selected from training examples
        double[] testValue3 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000238,-0.00129,-0.014578,-0.020715,-0.01944,-0.007742,0.000058,0.00007,0,0,0,0,0,0,0,0,0,0,0,0,-0.006307,0.020444,0.513928,0.677929,0.298813,0.056915,-0.008026,0.000021,0,0,0,0,0,0,0,0,0,0,0,0,-0.007176,0.036011,0.453742,0.832098,1.004046,0.557563,0.027653,-0.007432,0,0,0,0,0,0,0,0,0,0,0,0,0.000325,-0.00376,-0.012749,0.134303,0.753256,0.976497,0.120639,-0.018443,0,0,0,0,0,-0,-0,0,0,0,0,0,0.000031,0.000061,-0.006126,-0.019472,0.579224,1.049207,0.144629,-0.020939,0,0,0,0.000241,-0.001412,-0.016668,-0.010865,0.00049,0.000073,-0.000308,-0.00447,-0.002123,0.000131,0,-0.002664,-0.005319,0.58279,1.052438,0.145767,-0.021062,0,0,0,-0.000084,-0.01693,0.298567,0.460269,-0.004085,-0.001583,-0.006337,-0.002011,-0.011324,0.00064,0.000098,-0.0056,0.019079,0.819844,0.791951,0.064307,-0.012297,0,0,0.000187,-0.026739,0.188917,0.902936,0.43188,-0.022156,-0.012559,0.073476,0.491621,0.319049,-0.024066,-0.000353,-0.028202,0.304376,1.007257,0.534364,-0.016419,-0.003601,0,0,-0.002467,-0.025656,0.49501,0.92033,0.092737,-0.011055,-0.029162,0.406327,1.044801,0.75352,0.007596,-0.0456,0.183718,0.844726,0.940465,0.236931,-0.030711,0.00008,0,0,-0.011948,0.061037,0.786943,0.735692,0.005065,-0.012599,0.033774,0.682488,1.048655,0.924586,0.304997,0.283057,0.878936,1.00768,0.391209,-0.00703,-0.002232,0.000028,0,0,-0.014933,0.088879,0.875892,0.633767,-0.000041,-0.032064,0.4444,0.974917,0.606141,0.964741,0.904645,0.991414,0.958816,0.510526,0.020711,-0.006175,0.000256,0,0,0,-0.017701,0.114577,0.957316,0.580026,-0.04571,0.253835,0.944223,0.625163,0.02941,0.480822,0.920977,0.682414,0.229035,0.00001,-0.007604,0.00028,0,0,0,0,-0.017839,0.115849,0.95508,0.663875,0.283374,0.807259,0.90133,0.077443,-0.02009,-0.01047,0.112147,0.011312,-0.021737,-0.005614,0.000229,0.000021,0,0,0,0,-0.016207,0.099499,0.900587,0.982451,0.959932,0.980519,0.3237,-0.025662,0.000778,-0.000726,-0.011671,-0.002846,0.000332,0,0,0,0,0,0,0,-0.004371,0.000551,0.464333,0.889571,0.804037,0.317319,-0.022565,-0.003227,0.000031,0,0,0,0,0,0,0,0,0,0,0,0.000312,-0.007859,0.031322,0.150447,0.089407,-0.0297,-0.000707,0.000238,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000752,-0.00619,-0.020717,-0.013979,0.000843,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        for (int i = 1; i < 11; i++) {
            String file = "D:\\Projects-repos\\MachineLearningTest\\src\\com\\mkis\\assignments\\multiclassclassification\\data_" + i + ".txt";
            iterations = 0;
            dataSet.clear();
            meansOfFeatures.clear();
            maxMinusMinOfFeatures.clear();
            getNumberOfFeaturesAndTrainingExamples(file);
            loadData(file);
            initTheta();
            //System.out.println("Cost Function at start: " + createCostFunction(dataSet));
            doGradientDescent(dataSet);
            double[] featureScaledTestValue = createFeatureScaledTestValue(testValue3);
            //System.out.println("Probability that the number on the picture is a " + i +": " +nf.format(100 * (createHypothesis(featureScaledTestValue))) + " %");
            //System.out.println("Cost Function at the end: " + createCostFunction(dataSet));
            //System.out.println("Accuracy of the model: " + calcAccuracyOfModel(dataSet) + " %");
            //System.out.println("---------------------------------------------------------------------------------------");
            probabilities.put(i, createHypothesis(featureScaledTestValue));
        }
        System.out.println("The number on the picture is a: " +getTheMostProbableNumber());
    }

    //Nested class for the instances
    public static class Instance {
        double yValue;
        double[] xVariables;

        Instance(double yValue, double[] xVariables) {
            this.yValue = yValue;
            this.xVariables = xVariables;
        }
    }

    private static Integer getTheMostProbableNumber() {
        Map.Entry<Integer, Double> maxProbability = null;
        for (Map.Entry<Integer, Double> entry : probabilities.entrySet()) {
            if (maxProbability == null || entry.getValue().compareTo(maxProbability.getValue()) > 0) {
                maxProbability = entry;
            }
        }
        return maxProbability.getKey();
    }

    //Feature normalization of the test picture values
    private static double[] createFeatureScaledTestValue(double[] testValue) {
        double[] featureScaledTestValue = new double[n];
        featureScaledTestValue[0] = 1.0;
        for (int i = 0; i < n - 1; i++) {
            if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                featureScaledTestValue[i + 1] = testValue[i];
            } else {
                featureScaledTestValue[i + 1] = (testValue[i] - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i);
            }
        }
        return featureScaledTestValue;
    }

    //Also prepares for mean normalization
    private static void getNumberOfFeaturesAndTrainingExamples(String file) {
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
            //for mean normalization get means and (max-min)s of all all features
            for (int i = 0; i < n - 1; i++) {
                meansOfFeatures.add(i, tempList.get(i).stream().mapToDouble(val -> val).average().getAsDouble());
                maxMinusMinOfFeatures.add(i, Collections.max(tempList.get(i)) - Collections.min(tempList.get(i)));
                //System.out.println("Variable "+i+" Mean: " + meansOfFeatures.get(i) + " Max-min: " +maxMinusMinOfFeatures.get(i));
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Load the data from the txt file into an array
    private static void loadData(String file) {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            String[] columns;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                double y = Double.parseDouble(columns[n - 1]);
                double xArray[] = new double[n];
                xArray[0] = 1.0;
                for (int i = 0; i < n - 1; i++) {
                    if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                        xArray[i + 1] = Double.parseDouble(columns[i]);
                    } else {
                        // feature scaling applied, to get variables between 0 and 1
                        xArray[i + 1] = (Double.parseDouble(columns[i]) - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i);
                    }
                }
               /* //Input variables with 1 added to first column
                for (int i = 0; i < n - 1; i++) {
                    System.out.print(xArray[i] + " | ");
                }
                System.out.println();*/
                Instance instance = new Instance(y, xArray);
                dataSet.add(instance);
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Initialize theta with zeros
    private static void initTheta() {
        theta = new Double[n];
        for (int i = 0; i < n; i++) {
            theta[i] = 0.0;
        }
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
                    temp[i] += (hypothesis - y) * x[i];
                }
            }
            double alpha = 0.0001;
            for (int i = 0; i < n; i++) {
                theta[i] = theta[i] - alpha * temp[i];
            }
            iterations++;
            //Cost function to descend, theta after each iteration:
            //System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta) + ", Cost function: " + createCostFunction(dataSet));
            if (costFunctionOld - createCostFunction(dataSet) < 0.00001) {
                //System.out.println("Number of iterations: " + iterations);
                return;
            }
        }
        //System.out.println("Number of iterations: " + iterations);
    }

    //Predict whether the label is 0 or 1 (not admitted or admitted) using
    //a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    private static double predict(double[] scores) {
        if (createHypothesis(scores) >= 0.5) return 1;
        return 0;
    }

    //Calculate accuracy of logistic regression
    private static double calcAccuracyOfModel(List<Instance> instances) {
        int counter = 0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            if (predict(x) != y) counter++;
        }
        return (1 - counter / m) * 100;
    }

    //Created new txts for one vs all method
    private static void createTXTsForOneVsAll() throws java.lang.Exception {
        String path = "D:\\Projects-repos\\MachineLearningTest\\src\\com\\mkis\\assignments\\multiclassclassification\\data1.txt";
        for (int i = 1; i < 11; i++) {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));
            loader.setNoHeaderRowPresent(true);
            Instances dataSet = loader.getDataSet();
            int numberOfAttributes = dataSet.numAttributes() - 1;
            int numberOfInstances = dataSet.numInstances();
            for (int j = 0; j < numberOfInstances; j++) {
                if (dataSet.get(j).value(dataSet.attribute(numberOfAttributes)) == (double) i) {
                    dataSet.get(j).setValue(dataSet.attribute(numberOfAttributes), 1);
                } else {
                    dataSet.get(j).setValue(dataSet.attribute(numberOfAttributes), 0);
                }
            }
            CSVSaver saver = new CSVSaver();
            saver.setInstances(dataSet);
            saver.setFile(new File("D:\\Projects-repos\\MachineLearningTest\\src\\com\\mkis\\assignments\\multiclassclassification\\data_" + i + ".txt"));
            saver.writeBatch();
        }
    }

}
