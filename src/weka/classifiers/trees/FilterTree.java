package weka.classifiers.trees;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import weka.filters.AllFilter;
import weka.filters.Filter;
import java.io.Serializable;
import java.util.*;

public class FilterTree extends RandomizableClassifier {

    /**The root node of the decision tree**/
    protected Node m_RootNode;

    /**The filter to use locally at each node**/
    protected Filter m_Filter = new AllFilter();

    /**The minimum number of instances required for splitting**/
    protected double m_MinInstances = 1.0;//CHANGE TO 2.0

    /**A random number generator**/
    protected Random m_Random;

    @OptionMetadata(
            displayName = "threshold",
            description = "The minimum number of instances required for splitting (default = 2.0).",
            commandLineParamName = "M", commandLineParamSynopsis = "-M <double>",
            displayOrder = 1)
    public double getThreshold() {
        return m_MinInstances;
    }

    public void setThreshold(double threshold) {
        this.m_MinInstances = threshold;
    }

    @OptionMetadata(
            displayName = "filter",
            description = "The filter to use for splitting data, including filter options (default = AllFilter).",
            commandLineParamName = "F", commandLineParamSynopsis = "-F <filter specification>",
            displayOrder = 2)
    public Filter getFilter() {
        return m_Filter;
    }

    public void setFilter(Filter filter) {
        this.m_Filter = filter;
    }

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for building a classification tree with local filter models for definining splits.";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }

    /**
     * An interface indicating objects storing node information, implemented by three node info classes.
     */
    private interface NodeInfo extends Serializable {};

    /**
     * Class whose objects represent split nodes.
     */
    private class SplitNodeInfo implements NodeInfo {

        // The left node
        public Node Left = null;

        // The right node
        public Node Right = null;

        // The attribute used for splitting
        protected Attribute SplitAttribute;

        // The split value
        protected double SplitValue;

        //The filter used for the node
        protected Filter Filter;


        /**
         * Constructs a SplitNodeInfo object
         *
         * @param splitAttribute the attribute that defines the split
         * @param splitValue the value used for the split
         * @param filter the filter used at the node
         */
        public SplitNodeInfo(Attribute splitAttribute, double splitValue, Filter filter) {
            SplitAttribute = splitAttribute;
            SplitValue = splitValue;
            Filter = filter;
        }
    }

    /**
     * Class whose objects represent leaf nodes.
     */
    private class LeafNodeInfo implements NodeInfo {

        // The array of predictions
        protected double[] Prediction;

        /**
         * Constructs a LeafNodeInfo object.
         *
         * @param prediction the array of predictions to store at this node
         */
        public LeafNodeInfo(double[] prediction) {
            Prediction = prediction;
        }
    }

    /**
     * Class whose objects represent unexpanded nodes.
     */
    private class UnexpandedNodeInfo implements NodeInfo {

       protected Instances Instances;


        /**
         * Constructs an UnexpandedNodeInfo object.
         *
         * @param
         */
        public UnexpandedNodeInfo(Instances instances) {
            Instances = instances;
        }
    }

    /**
     * Class representing a node in the decision tree.
     */
    private class Node implements Serializable {

        // The node information object that stores the actual information for this node.
        protected NodeInfo NodeInfo;

        /**
         * Constructs a node based on the give node info.
         *
         * @param nodeInfo an appropriate node information object
         */
        public Node(FilterTree.NodeInfo nodeInfo) {
            NodeInfo = nodeInfo;
        }
    }



    protected Node makeLeafNode(Node node){

        Instances instances = ((UnexpandedNodeInfo)node.NodeInfo).Instances;

        double[] predictionOutput = new double[instances.numClasses()];

        //Calculating the class distribution at the node
        for(Instance i: instances){
            predictionOutput[(int)i.classValue()]++;
        }

        Utils.normalize(predictionOutput);
        node.NodeInfo = new LeafNodeInfo(predictionOutput);
        return node;

    }

    protected int[][] createSufficientStatistics(Instances newInstances){
        int[][] currentStats = new int[2][newInstances.numClasses()+1];

        //Calculating left side statistics
        currentStats[0][(int)newInstances.get(0).classValue()]++;
        currentStats[0][newInstances.numClasses()]++; //Amount of instances in left side statistics

        //Calculating the right side statistics
        for (int j = 1; j < newInstances.size() ; j++) {
            currentStats[1][(int)newInstances.get(j).classValue()]++;
            currentStats[1][newInstances.numClasses()]++;
        }

        return currentStats;
    }

    protected void updateSufficientStatistics(Instance instanceToMove,Instances newInstances, int[][] currentStats){

        //Removing from the right side of tree statistics
        currentStats[1][(int) instanceToMove.classValue()] --;
        currentStats[1][newInstances.numClasses()]--;

        //Add instance to the left side;
        currentStats[0][(int) instanceToMove.classValue()] ++;
        currentStats[0][newInstances.numClasses()]++;
    }

    protected double calculateEntropy(int[] inputStats){

        double finalVal = 0.0,out1,out2;
        for (int i = 0; i < inputStats.length - 1; i++) {
            out1 = Math.log((double) inputStats[i]/inputStats[inputStats.length - 1])/Math.log(2);
            out2 = (double) -inputStats[i]/inputStats[inputStats.length - 1];
            finalVal += out2 * out1;
        }

        //Returning Entropy for given probabilities
        return finalVal;

    }

    protected double calculateExpectedEntropyBeforeSplit(Instances instances){

        int[] classValueCount = new int[instances.numClasses()];
        double[] probValues = new double[instances.numClasses()];
        double[] entropyValues = new double[instances.numClasses()];
        double[] outputValues = new double[instances.numClasses()];
        int size = instances.size();
        double finalValue = 0.0;

        /*int[] classValueCount = new int[]{9,5};
        double[] probValues = new double[2];
        double[] entropyValues = new double[2];
        double[] outputValues = new double[2];
        int size = 14;*/

        for (Instance i: instances) {
            classValueCount[((int) i.classValue())]++;
        }

        for (int i = 0; i < classValueCount.length; i++) {
            probValues[i] = (double) classValueCount[i]/size;
            if(probValues[i] != 0.0){
                entropyValues[i] = Math.log((double) probValues[i])/Math.log(2);
                outputValues[i] -= (probValues[i] * entropyValues[i]);
            }
        }

        //Calculating expected entropy
        for (int i = 0; i < outputValues.length; i++) {
            finalValue += outputValues[i];
        }

        return finalValue;
    }

    protected double calculateExpectedEntropy(int[][] inputStats){

        //double[] classes = new double[inputStats[0].length - 1];
        double left = 0.0,out1Left,out2Left,probLeft;
        int totalLeft = inputStats[0][inputStats[0].length - 1];
        double right = 0.0,out1Right,out2Right,probRight;
        int totalRight = inputStats[1][inputStats[1].length - 1 ];
        int totalInstances = totalLeft + totalRight;

        for (int i = 0; i < inputStats[0].length - 1; i++) {

            //Calculate entropy left side
            probLeft = (double) inputStats[0][i]/totalLeft;
            //Making sure it is not zero or it will be NaN - Have to treat it as a zero for computational reasons
            if(probLeft != 0.0){
                out1Left = Math.log((double) probLeft)/Math.log(2);
                left -= (probLeft * out1Left);
            }

            //Calculate entropy right side
            probRight = (double) inputStats[1][i]/totalRight;
            //Making sure it is not zero or it will be NaN - Have to treat it as a zero for computational reasons
            if(probRight != 0.0){
                out1Right = Math.log((double) inputStats[1][i]/totalRight)/Math.log(2);
                right -= (probRight * out1Right);
            }

        }
        return ((double) totalLeft/totalInstances * left) + ((double) totalRight/totalInstances * right);
    }



    /**
     * Method that processes a node. Assumes that the given node is unexpanded. Turns the node
     * into a leaf node or split node as appropriate by replacing the node information.
     *
     * @param node the unexpanded node to process
     * @return the node with updated node information, turning it into a split node or leaf node
     */
    protected Node splitNode(Node node) throws Exception {

        UnexpandedNodeInfo newNode = ((UnexpandedNodeInfo)node.NodeInfo);

        int[][] currentStats;

        //Checking stopping criteria - Certain Number of instances met, as specified by the user
        if((newNode.Instances.size()) <= m_MinInstances){
            return makeLeafNode(node);
        }

        //Making a deep copy of the instances so they can be filtered
        //Instances instancesToFilter = new Instances(newNode.Instances);
        //Instances instancesToFilter = newNode.Instances;

        //Making a deep copy of the data
        Filter filter = Filter.makeCopy(m_Filter);

        //Allowing randomizable filter if is of type randomizable
        if (filter instanceof Randomizable) {
            ((Randomizable) filter).setSeed(m_Random.nextInt());
        }

        //Setting up input format of filter
        filter.setInputFormat(newNode.Instances);
        //Filtering the instances based on a filter specified by the user
        Instances newInstances = Filter.useFilter(newNode.Instances,filter);

        /*System.out.println("first:" + newNode.Instances.get(0));
        System.out.println("second: " + newInstances.get(0));
        System.out.println("first:" + newNode.Instances.get(2));
        System.out.println("second: " + newInstances.get(2));

        filter.input(newNode.Instances.get(2));
        filter.batchFinished();
        Instance instance1 = filter.output();

        System.out.println("first filter: " + newNode.Instances.get(2));
        System.out.println("second filter: " + instance1);*/

        double bestSplitValue = 0; //best Split value for the node
        double minExpectedEntropy = 0.0; //Best Expected entropy for a split -> want to minimise this
        double currentExpectedEntropy; //Current entropy for a split
        double newSplitValue; //Current split value
        Attribute bestAttribute = null;
        boolean lock = false;
        double entropyOfCurrentNode = calculateExpectedEntropyBeforeSplit(newInstances);

        //double out333 = calculateExpectedEntropyBeforeSplit(newInstances);

        //Iterating through the attributes
        for (int i = 0; i < newInstances.numAttributes(); i++) {

            newInstances.sort(i);//Sorting Attributes

            currentStats = createSufficientStatistics(newInstances);//Creating the current sufficient statistics

            double oldVal = newInstances.get(0).value(i);

            //Going through the attribute values and working out the split points
            for (int j = 1; j < newInstances.size(); j++) {

                currentExpectedEntropy = calculateExpectedEntropy(currentStats);

                if((currentExpectedEntropy < minExpectedEntropy) || (!lock)){

                    //Calculating Split Value
                    newSplitValue = (oldVal + newInstances.get(j).value(i))/2.0;

                    //If the old value and the new value are the same don't change
                    if(newSplitValue != oldVal){
                        minExpectedEntropy = currentExpectedEntropy;
                        bestAttribute = newInstances.attribute(i);
                        bestSplitValue = newSplitValue;
                        lock = true;//Setting lock so it can't get in the loop unless it meets first Criterion.
                    }
                }

                oldVal = newInstances.get(j).value(i);
                //Move sufficient statistics to the left and get the value
                updateSufficientStatistics(newInstances.get(j),newInstances,currentStats);
            }
        }

        //Check stop criterion again if information gain has not increased
        double informationGain = entropyOfCurrentNode - minExpectedEntropy;
        if(informationGain <= 0.0 || bestAttribute == null){
            return makeLeafNode(node);
        }

        //Splitting data into two subsets base on the filter value
        Instances[] subsets = new Instances[2];
        subsets[0] = new Instances(newNode.Instances, newNode.Instances.numInstances());
        subsets[1] = new Instances(newNode.Instances, newNode.Instances.numInstances());

        //CHECK THIS
        //Instance testInstance;

        //Iterating over the instance and determining the correct subset
        /*for (Instance instance : newNode.Instances) {

            //Applying filter to work out the best split
            filter.input(instance);
            testInstance = filter.output();

            subsets[testInstance.value(bestAttribute) < bestSplitValue ? 0 : 1].add(instance);
        }*/

        //Calculating what subset to send instance into
        for (int i = 0; i < newNode.Instances.size(); i++) {
            subsets[newInstances.get(i).value(bestAttribute) < bestSplitValue ? 0 : 1].add(newNode.Instances.get(i));
        }

        //Transforming node into a split node
        node.NodeInfo = new SplitNodeInfo(bestAttribute,bestSplitValue,filter);

        //Process left side of tree
        ((SplitNodeInfo)node.NodeInfo).Left = splitNode(new Node(new UnexpandedNodeInfo(subsets[0])));

        //process right side of tree
        ((SplitNodeInfo)node.NodeInfo).Right = splitNode(new Node(new UnexpandedNodeInfo(subsets[1])));

        return node;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        //Setting random seed of random object
        m_Random = instances.getRandomNumberGenerator(getSeed());

        //Creating rootNode
        Node newNode = new Node(new UnexpandedNodeInfo(instances));

        //Processing Node
        m_RootNode = splitNode(newNode);


    }

    private void travTree(Node node){

        //Getting the root node to start tree traversal
        Node currNode = node;

        if(currNode.NodeInfo instanceof LeafNodeInfo){
            System.out.println("leafNode");
            System.out.println(((LeafNodeInfo)currNode.NodeInfo).Prediction);
        }

        System.out.println("left-----------");
        travTree(((SplitNodeInfo)currNode.NodeInfo).Left);
        System.out.println("left-----------");
        System.out.println("right----------");
        travTree(((SplitNodeInfo)currNode.NodeInfo).Right);
        System.out.println("right----------");

    }

    /*@Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return super.distributionsForInstances(batch);
    }*/


    public double[] distributionsForInstances(Instance instance) throws Exception {

        double[] hllo = new double[]{0,1,0};
        return hllo;

        //Getting the root node to start tree traversal
        /*Node currNode = m_RootNode;

        //If root is leaf node return prediction
        if(currNode.NodeInfo instanceof LeafNodeInfo){
            return ((LeafNodeInfo)currNode.NodeInfo).Prediction;
        }

        //Traversing the tree
        return traverseTree(currNode,instance);*/
    }

    protected double[] traverseTree(Node node,Instance instance) throws Exception {

        Instance filteredInstance;
        Filter currentNodeFilter;
        double[] pred = null;



        SplitNodeInfo currNode = ((SplitNodeInfo) node.NodeInfo);

        //Getting filter to use on the instance values
        currentNodeFilter = currNode.Filter;

        //Filtering the instance to be checked
        currentNodeFilter.input(instance);
        filteredInstance = currentNodeFilter.output();

        if(filteredInstance.value(currNode.SplitAttribute) < currNode.SplitValue){
            //Traversing down left branch
            return traverseTree(currNode.Left,instance);
        }
        else{
            //Traversing down right branch
            return traverseTree(currNode.Right,instance);

        }
    }



    /*public boolean implementsMoreEfficientBatchPrediction() {
        return true;
    }*/

    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runClassifier(new FilterTree(), options);
    }
}
