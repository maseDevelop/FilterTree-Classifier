package weka.classifiers.meta;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import weka.filters.AllFilter;
import weka.filters.Filter;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class FilterTree extends RandomizableClassifier {

    /**The root node of the decision tree**/
    protected Node m_RootNode;

    /**The filter to use locally at each node**/
    protected Filter m_Filter = new AllFilter();

    /**The minimum number of instances required for splitting**/
    protected double m_MinInstances = 2.0;

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
        return "Class for building a classification tree with local filter models for defining splits.";
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

        //Stores class count for toString tree output
        protected double[] ClassCountForString;

        /**
         * Constructs a LeafNodeInfo object.
         *
         * @param prediction the array of predictions to store at this node
         * @param classCount the count of each class
         */
        public LeafNodeInfo(double[] prediction, double[] classCount) {
            Prediction = prediction;
            ClassCountForString = classCount;
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


    /**
     * Constructs a leaf node from an unexpanded node
     *
     * @param node that is turned into a leaf node
     * @return the newly constructed leaf node with the relevant prediction data
     */
    protected Node makeLeafNode(Node node){

        Instances instances = ((UnexpandedNodeInfo)node.NodeInfo).Instances;

        double[] predictionOutput = new double[instances.numClasses()];
        double[] classCount = new double[instances.numClasses()];

        //Calculating the class distribution at the node
        for(Instance i: instances){
            predictionOutput[(int)i.classValue()]++;
            classCount[(int)i.classValue()]++;
        }
        Utils.normalize(predictionOutput);
        node.NodeInfo = new LeafNodeInfo(predictionOutput,classCount);
        return node;

    }

    /**
     * Creates a sufficient statistics 2D array based on the instances given to the method
     * it stores the class values and the count of the class values each side of a binary split
     * @param newInstances is what is used to create the sufficient statistics array
     * @return 2D array of size to corresponding to class values either side of a binary split
     */
    protected int[][] createSufficientStatistics(Instances newInstances){
        int[][] currentStats = new int[2][newInstances.numClasses()+1];

        //Calculating left side statistics, initially this will be the first value
        currentStats[0][(int)newInstances.get(0).classValue()]++;
        currentStats[0][newInstances.numClasses()]++; //Amount of instances in left side of tree

        //Calculating the right side statistics
        for (int j = 1; j < newInstances.size() ; j++) {
            currentStats[1][(int)newInstances.get(j).classValue()]++;
            currentStats[1][newInstances.numClasses()]++;//Amount of instances on the right side of the tree
        }

        return currentStats;
    }

    /**
     * Moves up the sufficient statistics array from left to right,
     * each time placing one class value from the right side on the left
     * @param instanceToMove instance to move from right side of the split to left side of the split
     * @param newInstances the instances to interact with
     * @param currentStats the current Sufficient statistics array
     */
    protected void updateSufficientStatistics(Instance instanceToMove,Instances newInstances, int[][] currentStats){

        //Removing from the right side of tree statistics
        currentStats[1][(int) instanceToMove.classValue()] --;
        currentStats[1][newInstances.numClasses()]--;

        //Add instance to the left side;
        currentStats[0][(int) instanceToMove.classValue()] ++;
        currentStats[0][newInstances.numClasses()]++;
    }

    /*protected double calculateEntropy(int[] inputStats){

        double finalVal = 0.0,out1,out2;
        for (int i = 0; i < inputStats.length - 1; i++) {
            out1 = Math.log((double) inputStats[i]/inputStats[inputStats.length - 1])/Math.log(2);
            out2 = (double) -inputStats[i]/inputStats[inputStats.length - 1];
            finalVal += out2 * out1;
        }

        //Returning Entropy for given probabilities
        return finalVal;

    }*/

    /**
     * Used to calculate the entropy of a node before a split
     *
     * @param instances the instance of a node to calculate the expected entropy
     * @return a value of entropy for the instance given
     */
    protected double calculateExpectedEntropyBeforeSplit(Instances instances){

        int[] classValueCount = new int[instances.numClasses()];
        double[] probValues = new double[instances.numClasses()];
        double[] entropyValues = new double[instances.numClasses()];
        double[] outputValues = new double[instances.numClasses()];
        int size = instances.size();
        double finalValue = 0.0;

        //getting a class value count
        for (Instance i: instances) {
            classValueCount[((int) i.classValue())]++;
        }

        //Calculating entropy of each class
        for (int i = 0; i < classValueCount.length; i++) {
            probValues[i] = (double) classValueCount[i]/size;
            if(probValues[i] != 0.0){
                entropyValues[i] = Math.log((double) probValues[i])/Math.log(2);
                outputValues[i] -= (probValues[i] * entropyValues[i]);
            }
        }

        //Calculating entropy for all classes
        for (int i = 0; i < outputValues.length; i++) {
            finalValue += outputValues[i];
        }

        return finalValue;
    }

    /**
     * Used to calculate expected entropy for a certain binary split
     *
     * @param inputStats the input sufficent statistics
     * @return a entropy value for the given statistics
     */
    protected double calculateExpectedEntropy(int[][] inputStats){

        double left = 0.0,out1Left,probLeft;
        int totalLeft = inputStats[0][inputStats[0].length - 1];
        double right = 0.0,out1Right,probRight;
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

        //Creating new a copy of the instances as when they get sorted it messes up the index pairing
        Instances FilteredInstances = new Instances(newInstances);

        double bestSplitValue = 0; //best Split value for the node
        double minExpectedEntropy = 0.0; //Best Expected entropy for a split -> want to minimise this
        double currentExpectedEntropy; //Current entropy for a split
        double newSplitValue; //Current split value
        Attribute bestAttribute = null;
        boolean lock = false;

        //Calculating the entropy of the node - used to calculate information gain
        double entropyOfCurrentNode = calculateExpectedEntropyBeforeSplit(newInstances);

        //Iterating through the attributes
        for (int i = 0; i < newInstances.numAttributes() - 1; i++) {

            newInstances.sort(i);//Sorting Attributes

            currentStats = createSufficientStatistics(newInstances);//Creating the current sufficient statistics

            double oldVal = newInstances.get(0).value(i);

            //Going through the attribute values and working out the split points
            for (int j = 1; j < newInstances.size(); j++) {

                //Calculating current expected entropy based on the current sufficient statistics
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

        //Calculating what subset to send instance into
        for (int i = 0; i < newNode.Instances.size(); i++) {
            subsets[FilteredInstances.get(i).value(bestAttribute) < bestSplitValue ? 0 : 1].add(newNode.Instances.get(i));
        }


        //Transforming node into a split node
        node.NodeInfo = new SplitNodeInfo(bestAttribute,bestSplitValue,filter);

        //Clean up
        bestAttribute = null;
        FilteredInstances = null;
        newInstances = null;
        filter = null;
        currentStats = null;
        newNode = null;

        //Process left side of tree
        ((SplitNodeInfo)node.NodeInfo).Left = splitNode(new Node(new UnexpandedNodeInfo(subsets[0])));

        //process right side of tree
        ((SplitNodeInfo)node.NodeInfo).Right = splitNode(new Node(new UnexpandedNodeInfo(subsets[1])));

        return node;

    }

    /**
     * Builds the classifier
     * @param instances that are used to build the classifier
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {

        //Setting random seed of random object
        m_Random = instances.getRandomNumberGenerator(getSeed());

        //Creating rootNode
        Node newNode = new Node(new UnexpandedNodeInfo(instances));

        //Processing Node
        m_RootNode = splitNode(newNode);
    }

    /*@Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {

        //If root is leaf node return prediction
        if(m_RootNode.NodeInfo instanceof LeafNodeInfo){
            double[][] outArray = new double[batch.size()][];
            for (int i = 0; i < batch.size(); i++) {
                outArray[i] = ((LeafNodeInfo)m_RootNode.NodeInfo).Prediction;
            }
            return outArray;
        }

        int[] out = IntStream.range(0, batch.size()).toArray();
        double[][] batchPredictions = new double[batch.size()][];

        //Traversing the tree
        traverseTreeBatch(m_RootNode,out,batch,batchPredictions);

        return batchPredictions;
    }

    @Override
    public boolean implementsMoreEfficientBatchPrediction() {
        return true;
    }

    protected void traverseTreeBatch(Node node, int[] indexes, Instances batch, double[][] predictionsArray) throws Exception {

        double[][] outArr = new double[indexes.length][];

        //If root is leaf node return prediction
        if(node.NodeInfo instanceof LeafNodeInfo){

            for (int i = 0; i < indexes.length; i++) {
                predictionsArray[indexes[i]] = ((LeafNodeInfo)node.NodeInfo).Prediction;
            }
        }

        //It is a splitNode as leaf node should have been returned, so it is safe to cast
        SplitNodeInfo currNode = ((SplitNodeInfo) node.NodeInfo);

        //Setting up input format of filter
        currNode.Filter.setInputFormat(batch);
        //Filtering the instances based on a filter specified by the user
        Instances FilteredInstances = Filter.useFilter(batch,currNode.Filter);

        //Splitting data into two subsets base on the filter value
        Instances[] subsets = new Instances[2];
        subsets[0] = new Instances(batch, batch.numInstances());
        subsets[1] = new Instances(batch, batch.numInstances());

        HashMap<Integer,ArrayList<Integer>> intSubsets = new HashMap<>();


        int branch;
        //Calculating what subset to send instance into
        for (int i = 0; i < batch.size(); i++) {
            branch = FilteredInstances.get(i).value(currNode.SplitAttribute) < currNode.SplitValue ? 0 : 1;
            subsets[branch].add(batch.get(i));
            intSubsets.get(branch).add(i);
        }


        //Going down the left side
        traverseTreeBatch(currNode.Left,intSubsets.get(0).stream().mapToInt(Integer::intValue).toArray(), subsets[0],predictionsArray);
        //Going down the right side
        traverseTreeBatch(currNode.Right,intSubsets.get(1).stream().mapToInt(Integer::intValue).toArray(), subsets[1],predictionsArray);
    }*/


    /**
     * Gets the class distribution for an instance
     * @param instance the instance you want to find a prediction for
     * @return array of based on class distribution of the node
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        //Getting the root node to start tree traversal
        Node currNode = m_RootNode;

        //If root is leaf node return prediction
        if(currNode.NodeInfo instanceof LeafNodeInfo){
            return ((LeafNodeInfo)currNode.NodeInfo).Prediction;
        }

        //Traversing the tree
        return traverseTreeInstance(currNode,instance);
    }


    /**
     * Recursively traverses a tree in order to get a leaf node
     * @param node the node to search for a prediction
     * @param instance the instance value to search for a prediction with
     * @return a prediction based on the class distribution from a leaf node
     * @throws Exception
     */
    protected double[] traverseTreeInstance(Node node,Instance instance) throws Exception {

        Instance filteredInstance;
        Filter currentNodeFilter;

        if(node.NodeInfo instanceof LeafNodeInfo){
            return ((LeafNodeInfo)node.NodeInfo).Prediction;
        }

        //It is a splitNode as leaf node should have been returned, so it is safe to cast
        SplitNodeInfo currNode = ((SplitNodeInfo) node.NodeInfo);

        //Getting filter to use on the instance values
        currentNodeFilter = currNode.Filter;

        //Filtering the instance to be checked
        currentNodeFilter.input(instance);
        currentNodeFilter.batchFinished();
        filteredInstance = currentNodeFilter.output();

        if(filteredInstance.value(currNode.SplitAttribute) < currNode.SplitValue){
            //Traversing down left branch
            return traverseTreeInstance(currNode.Left,instance);
        }
        else{
            //Traversing down right branch
            return traverseTreeInstance(currNode.Right,instance);
        }
    }

    /**
     * Method that returns a textual description of the subtree attached to the given node. The description is
     * returned in a string buffer.
     *
     * @param stringBuffer buffer to hold the description
     * @param node the node whose subtree is to be described
     * @param levelString the level of the node in the overall tree structure
     */
    protected void toString(StringBuffer stringBuffer, Node node, String levelString) {

        if (node.NodeInfo instanceof SplitNodeInfo) {

            stringBuffer.append("\n" + levelString + ((SplitNodeInfo) node.NodeInfo).SplitAttribute.name() + " < " + Utils.doubleToString(((SplitNodeInfo) node.NodeInfo).SplitValue, getNumDecimalPlaces()));
            toString(stringBuffer, ((SplitNodeInfo) node.NodeInfo).Left, levelString + "|   ");
            stringBuffer.append("\n" + levelString + ((SplitNodeInfo) node.NodeInfo).SplitAttribute.name() + " >= " +
                    Utils.doubleToString(((SplitNodeInfo) node.NodeInfo).SplitValue, getNumDecimalPlaces()));
            toString(stringBuffer, ((SplitNodeInfo) node.NodeInfo).Right, levelString + "|   ");
        } else {
            double[] dist = ((LeafNodeInfo) node.NodeInfo).ClassCountForString;
            stringBuffer.append(":");
            for (double pred : dist) {
                stringBuffer.append(" " + Utils.doubleToString(pred, getNumDecimalPlaces()));
            }
        }
    }

    /**
     * Method that returns a textual description of the classifier.
     *
     * @return the textual description as a string
     */
    public String toString() {
        if (m_RootNode == null) {
            return "FilterTree: has not been built yet";
        }
        StringBuffer stringBuffer = new StringBuffer();
        toString(stringBuffer, m_RootNode, "");
        return stringBuffer.toString();
    }

    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runClassifier(new FilterTree(), options);
    }
}