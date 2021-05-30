package weka.classifiers.trees;


import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

public class FilterTree extends AbstractClassifier {

    /** for serialization */
    //static final long serialVersionUID = -251831442047263433L;

    /** The filter to use. */
    protected Filter m_Filter = new Randomize();

    /** The minimum number of instances used to continue growing the tree */
    protected int m_minInstances = 1;

    /** The root node of the tree**/
    protected Node rootNode;

    /** A list of instances that can be referenced for retrievial of an instance **/
    protected Instances GlobalInstances;

    /** Store computated sorted index of features **/
    protected int sortedFeaturesIndex[][];

    /** Stores class values of instances based on sortFeaturesIndex **/
    protected double sortedFeatureClassValue[][];

    /**
     * Returns the Capabilities of this classifier.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        //result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;


    }

    /** Handling the Filter parameter. */
    @OptionMetadata(
            displayName = "Filter function",
            description = "The Filter to use.", displayOrder = 1,
            commandLineParamName = "F",
            commandLineParamSynopsis = "-F <Filter specification>")
    public Filter getFilter() {  return m_Filter; }
    public void setFilter(Filter value) { m_Filter = value; }

    /** Handling the parameter setting the sample size. */
    @OptionMetadata(
            displayName = "The expansion threshold.",
            description = "The minimum number of instances used to continue growing the tree.", displayOrder = 3,
            commandLineParamName = "M",
            commandLineParamSynopsis = "-M <int>")
    public void setExpansionThreshold(int minNumberInstances) { m_minInstances = minNumberInstances; }
    public int getExpansionThreshold() { return m_minInstances; }

    /**
     * An interface indicating objects storing node information, implemented by three node info classes.
     */
    private interface NodeInfo extends Serializable {};

    /**
     * Class whose objects represent split nodes.
     */
    private class SplitNodeInfo implements NodeInfo {

        // The left node
        private Node left;

        // The right node
        private Node right;

        // The attribute used for splitting
        protected Attribute SplitAttribute;

        // The split value
        protected double SplitValue;

        //Getter for left
        protected Node getLeft() {return left;}

        //Setter for left
        protected void setLeft(Node node) {
            this.left = node;
            Instances[0] = node;
        }

        //Getter for right
        protected Node getRight() {return right;}

        //Setter for left
        protected void setRight(Node node) {
            this.right = node;
            Instances[1] = node;
        }

        //Pair of instances
        protected Node[] Instances;

        /**
         * Constructs a SplitNodeInfo object
         *
         * @param splitAttribute the attribute that defines the split
         * @param splitValue the value used for the split
         * @param left access to the left side of the tree
         * @param right access to the right side of the tree
         */
        public SplitNodeInfo(Attribute splitAttribute, double splitValue, Node left, Node right) {
            SplitAttribute = splitAttribute;
            SplitValue = splitValue;
            this.left = left;
            this.right = right;
            Instances = new Node[]{left,right};
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

        protected int StartIndex;

        protected int EndIndex;


        /**
         * Constructs an UnexpandedNodeInfo object.
         *
         * @param
         */
        public UnexpandedNodeInfo(int startIndex,int endIndex) {
            StartIndex = startIndex;
            EndIndex = endIndex;
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

        return null;
    }

    /**
     * Method that processes a node. Assumes that the given node is unexpanded. Turns the node
     * into a leaf node or split node as appropriate by replacing the node information.
     *
     * @param node the unexpanded node to process
     * @return the node with updated node information, turning it into a split node or leaf node
     */
    protected Node processNode(Node node) {

        UnexpandedNodeInfo newNode = ((UnexpandedNodeInfo)node.NodeInfo);
        Instance currInstance;

        int amountOfInstancesInNode = newNode.EndIndex - newNode.StartIndex;

        int[][] currentStats;


        //Checking stopping criteria - Certain Number of instances met, as specified by the user
        if((amountOfInstancesInNode + 1) <= m_minInstances){
            return makeLeafNode(node);
        }

        Attribute bestAttribute;
        double bestSplitValue;
        double maxEntropy;

        //System.out.println(GlobalInstances.get(sortedFeaturesIndex[0][newNode.StartIndex]).classValue());

        //CHECK THE >= OR > ON THE FOR LOOPS

        //Evaluating the split for each attribute
        for (int i = 0; i < sortedFeaturesIndex.length ; i++) {

            currentStats = new int[2][GlobalInstances.numClasses() + 1];//Stats for a binary split hence the size 2 [0,0,...n_classes,amount_of_instances]

            //Calculating left side statistics
            currentStats[0][(int)sortedFeatureClassValue[i][newNode.StartIndex]] = 1; //Class value
            currentStats[0][GlobalInstances.numClasses()] = 1; //Amount of instances in left side statistics

            //Calculating right side statistics
            for (int j = newNode.StartIndex + 1; j <= newNode.EndIndex ; j++) {
                currentStats[1][(int)sortedFeatureClassValue[i][j]]++; //Class value
                currentStats[1][GlobalInstances.numClasses()]++; //Amount of instances in left side statistics
            }

            //Calculate expected entropy based on all instances in the node.


            //Check if it is is the max




        }


        return null;

    }

    public String globalInfo() { return "A tree classifier that local applies a filter at each node."; }


    //Move this method
    private void setup(Instances instances){

        //Setting the list of global instances
        GlobalInstances = instances;
        //Creating sorted Attribute Index;
        int numAttributes = GlobalInstances.numAttributes()  - 1;
        int numInstances = GlobalInstances.numInstances() - 1 ;

        //Setting global variable
        sortedFeaturesIndex = new int[numAttributes][numInstances];
        sortedFeatureClassValue = new double[numAttributes][numInstances];

        for (int i = 0; i < numAttributes; i++) {
            //Need to initialise a new array each time as it will not work in lambda expression
            double[] finalFeatureValues = instances.attributeToDoubleArray(i);
            sortedFeaturesIndex[i] = IntStream.range(0,finalFeatureValues.length)
                    .boxed()
                    .sorted((a,b) -> {
                        if(finalFeatureValues[a] < finalFeatureValues[b]){return -1;}
                        else if (finalFeatureValues[a] > finalFeatureValues[b]){return 1;}
                        else {return 0;}
                    })
                    .mapToInt(element -> element)
                    .toArray();

            //This may not work with numerical attributes!!!!!!!
            sortedFeatureClassValue[i] = Arrays.stream(sortedFeaturesIndex[i])
                    .map((n) -> (int) GlobalInstances.get(n).classValue())
                    .mapToDouble(element -> element)
                    .toArray();
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        //Remove missing values

        //Setting up initial arrays - called once at when building classifier.
        setup(instances);

        //Creating a node list that has all tree nodes in it before they are finalised.
        ArrayList<Node> nodesInTree = new ArrayList<>();

        //Creating rootNode
        rootNode = new Node(new UnexpandedNodeInfo(0,(instances.size()-1))); //May need to change this given size considerations

        //Adding node to tree
        nodesInTree.add(rootNode);

        //Splitting node
        while(!nodesInTree.isEmpty()){
            Node newNode = processNode(rootNode);

            //Iterating through node subtrees and expanding to ethier split node or leafnode
            if(newNode.NodeInfo instanceof SplitNodeInfo){
                nodesInTree.addAll(Arrays.asList(((SplitNodeInfo) newNode.NodeInfo).Instances));
            }
        }
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return super.distributionsForInstances(batch);
    }

    public double[] distributionsForInstances(Instance instance) throws Exception {
        return new double[]{0};
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

   /*private class InstanceObj implements Comparator<InstanceObj>{

        int Index;
        double Value;

        public InstanceObj(int index,double value){
            super();
            Index = index;
            Value = value;
        }


        @Override
        public int compare(InstanceObj o1, InstanceObj o2) {

            if(o1.Value < o2.Value){
                return -1;
            }
            else if(o1.Value > o2.Value){
                return 1;
            }
            else{
                return 0;
            }
        }
    }
    //double[] featureValues;
        //InstanceObj[] objArr = new InstanceObj[numInstances];
           /*for (int i = 0; i < numAttributes-1; i++) {
            featureValues = GlobalInstances.attributeToDoubleArray(i);
            for (int j = 0; j < numInstances-1; j++) {
                objArr[j] = new InstanceObj(j,featureValues[j]);
            }

            objArr[0].compare(objArr[0],objArr[1]);

            Arrays.sort(objArr,0,objArr.length - 1);
            sortedFeaturesIndex[i] = Arrays.stream(objArr).mapToInt(n -> n.Index).toArray();


        }*/

