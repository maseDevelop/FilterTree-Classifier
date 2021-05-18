package weka.classifiers.trees;


import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.Serializable;

public class FilterTree extends AbstractClassifier {

    /** for serialization */
    //static final long serialVersionUID = -251831442047263433L;

    /** The kernel function to use. */
    protected Filter m_Filter = new Randomize();

    /** The minimum number of instances used to continue growing the tree */
    protected int m_minInstances = 100;

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
    protected interface NodeInfo extends Serializable {};

    /**
     * Class whose objects represent split nodes.
     */
    protected class SplitNodeInfo implements NodeInfo {

        // The attribute used for splitting
        protected Attribute SplitAttribute;

        // The split value
        protected double SplitValue;

        // The array of successor nodes
        protected Node[] Successors;

        /**
         * Constructs a SplitNodeInfo object
         *
         * @param splitAttribute the attribute that defines the split
         * @param splitValue the value used for the split
         * @param successors the array of successor nodes
         */
        public SplitNodeInfo(Attribute splitAttribute, double splitValue, Node[] successors) {
            SplitAttribute = splitAttribute;
            SplitValue = splitValue;
            Successors = successors;
        }
    }

    /**
     * Class whose objects represent leaf nodes.
     */
    protected class LeafNodeInfo implements NodeInfo {

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
    protected class UnexpandedNodeInfo implements NodeInfo {

        // The data to be used for expanding the node.
        protected Instances Data;

        /**
         * Constructs an UnexpandedNodeInfo object.
         *
         * @param data the data to be used for turning this node into an expanded node.
         */
        public UnexpandedNodeInfo(Instances data) {
            Data = data;
        }
    }

    /**
     * Class representing a node in the decision tree.
     */
    protected class Node implements Serializable {

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
     * Method that processes a node. Assumes that the given node is unexpanded. Turns the node
     * into a leaf node or split node as appropriate by replacing the node information.
     *
     * @param node the unexpanded node to process
     * @return the node with updated node information, turning it into a split node or leaf node
     */
    protected Node processNode(Node node) {
        return null;
    }



    public String globalInfo() { return "A tree classifier that local applies a filter at each node."; }



    @Override
    public void buildClassifier(Instances instances) throws Exception {

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
