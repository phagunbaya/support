.. _faq:

Frequently asked Questions
==========================


|   :ref:`Getting Started<getting_started>`
|   :ref:`Accounts<accounts>`
|   :ref:`Datastreams<datastreams>`
|   :ref:`Assessments<assessments>`
|   :ref:`Models<models>`
|   :ref:`Facts<facts>`
|   :ref:`Live Monitoring<live_monitoring>`
|   :ref:`Support<support>`
|   :ref:`Software Updates<software_updates>`
|   :ref:`Infrastructure and Deployment<infrastructure>`
   
   
.. _getting_started:




Falkonry - Getting started
---------------------------

**1. What is the relationship between Datastream, Assessment, Model, Facts, Conditions ?**

Please refer to the following link for an overview of Falkonry - Definition of Terms:

    `Falkonry Definition of Terms <http://help.falkonry.com/en/latest/conceptsoverview.html>`_



**2. What is an overall approach to use Falkonry?**

    #. Reach out to Falkonry to evaluate approach, use case and the possibility of starting a jumpstart evaluation at info@falkonry.com

    #. Falkonry creates a trial jumpstart account on its cloud environment

    #. Prepare use case data (select inut signals over a time range that well characterizes episodes and activities of interest) for ingestion into Falkonry

    #. Confirm approach: use Sliding windows vs batch Windows and the goal of the assessment output

    #. Create a Falkonry datastream by uploading the dataset

    #. Run unsupervised model 

    #. Add facts (ground truths) to establish conditions in Falkonry based on SME domain knowledge

    #. Train model using injected facts (Supervised model)

    #. Tweak and fine tune model using an iterative approach of facts injestion, and model parameter tuning

    #. Ingest additional data to the datastream for model validation

    #. When a satisfactory model is trained, identify live streaming data source for real time assessments

    #. Leverage Falkonry connectors, agents, ADK to stream data into Falkonry and tag back assessmenht output results back into your data historian/ HMI  

    #. Work towards an on-premise deployment model

    

.. _accounts:




Account
--------

**1. How many datastreams can be created within an account ?**

By default, when a new jumpstart account is provisioned the limitation for number of datastreams is set to 5.
Consult that table below for defaults:

+-----------------------------------------+---------+
|  Data storage limit                     |  100MB  |
+-----------------------------------------+---------+
|  Max number of Datastreams              |    5    |
+-----------------------------------------+---------+
|  Max number of parallel model creation  |    3    |
+-----------------------------------------+---------+
|  Assessment production limit            |    1    |
+-----------------------------------------+---------+

To change the default provisioning please reach out to Falkonry Admin at support[at]falkonry[dot]com



.. _datastreams:




Datastreams
-----------

**1. What are the various methods through which I can import data into Falkonry?**

     Data can be imported using one of the following:

     a. Directly from Falkonry LRS using the UI
     b. For OSIsoft PI customers, data can be ingested using the PI Agent, refer to :ref:`OSIsoft PI agent<pi_agent>` for more details
     c. Datastreams can be created using any of the following Falkonry provided ADKs
     	- Python
	    - Java
	    - C#
	    - Shell / bash
	 Detailed Falkonry ADK doumentation can be found at `Falkonry's ADK reference <adk_documentation.html>`_



**2. What is the frequency at which live data gets injected into Falkonry from the PI AF system or any other database? Is this setting configurable?**

     Falkonry PI agent can support live data ingestion sampling rates as aggressive as 500ms. 
     This is easily configurable using the Falkomnry PI Agent by using the Advanced Settings.  



**3. What is the maximum limit to the data that can be ingested into Falkonry?**
     
     Data ingestion limit is detemined based on your account settings. These can be changed by reaching out to Falkonry Admin at support[at]falkonry[dot]com



**4. What is the difference between Numerical and Categorical signal data?**
     
     *Numerical* - As the name suggests this signal has numerical data type and is suggestive of continuous data (for e.g. sensors, control systems, etc)
     *Categorical* - These are signals with discrete values that often suggest the state of a system or a primitive (e.g. True/ False, High/ Med/ Low, 0/1)



**5. What timestamp granularity does Falkonry support? (For example does it support time granularity finer than millisecond?)**

     Falkonry supports granularity down to microseconds



**7. Is there a common format and MIME type that is supported for ingesting CSV format data into a Falkonry datastream?**
     
     Yes. We support RFC 4180 regarding support for common format. More details can be found at `RFC <https://tools.ietf.org/html/rfc4180>`_
     Following is a list of implementations and formats that is supported in conjunction with RFC 4180.

     i.  Each record is located on a separate line, delimited by a linebreak (CRLF).  For example:

                aaa,bbb,ccc CRLF
                zzz,yyy,xxx CRLF

     ii.  The last record in the file may or may not have an ending line break.  For example:

                aaa,bbb,ccc CRLF
                zzz,yyy,xxx

     iii.  There maybe an optional header line appearing as the first line of the file with the same format as normal record lines. 
           This header will contain names corresponding to the fields in the file and should contain the same number of fields as the records in
           the rest of the file (the presence or absence of the header line should be indicated via the optional "header" parameter of this MIME type).  
           For example:

                field_name,field_name,field_name CRLF
                aaa,bbb,ccc CRLF
                zzz,yyy,xxx CRLF

     iv. Within the header and each record, there may be one or more fields, separated by commas.  Each line should contain the same
         number of fields throughout the file.  Spaces are considered part of a field and should not be ignored.  The last field in the
         record must not be followed by a comma.  For example:

                aaa,bbb,ccc

     v.  Each field may or may not be enclosed in double quotes (however some programs, such as Microsoft Excel, do not use double quotes at all).  
         If fields are not enclosed with double quotes, then double quotes may not appear inside the fields.  For example:

                "aaa","bbb","ccc" CRLF
                zzz,yyy,xxx

     vi.  Fields containing line breaks (CRLF), double quotes, and commas should be enclosed in double-quotes.  For example:

                "aaa","b CRLF
                bb","ccc" CRLF
                zzz,yyy,xxx

     vii.  If double-quotes are used to enclose fields, then a double-quote appearing inside a field must be escaped by preceding it with
           another double quote.  For example:

                "aaa","b""bb","ccc"


**8. What is the data file size limit when uploading to a datastream?**

    Falkonry can ingest reasonably large JSON/CSV data files successfully.
    However, it is always a good idea to start with smaller datasets so as to be able to iterate over model design and data schema.

    Once you have some reasonable confidence on the data schema, you may always update more data by accessing the Datastream Configuration window which is available at the top of the timeline UI

    .. image:: images/datastream_configuration.png

    Clicking on Upload Data will allow you to add more data at any later point in time.

    .. image:: images/upload_more_data.png


.. _assessments:



Assessments
-----------

**1. Are assessment results produced at frequency lesser than the database frequency?**

     Assessment results can be produced at any of the following 3 rates:

     a. System determined
        Falkonry determines the best assessment freqency based on input signal data and as needed to identify different conditions

     b. Mimimum interval
        Provide assessments as frequently as the input data allows

     c. Explicit user defined interval



**2. Under what circumstances do I use the Falkonry output as input for Modeling?**

    In certain continuous/ discrete operations setups, such as a manufacturing lines, the assessment output from one entity/ machine/ line can feed into further downstream operations.

    In such cases, it may help to introduce the output from an assesssment into another assessment within the same datastream by treating the assessment output as a categorical input signal for downstream assessments.

    Falkonry makes it easy to work with these "chained assessments" by allowing a user to treat the output of an assessment as an input signal for training model for downstream assessments.

    Simply select the assessment output from another assessment when creating a model in the Model panel as shown:

    .. image:: images/chained_assessment.png


.. _models:




Models
------
**1. What is a Sliding Window?**

    In Sliding windows, incoming signals may not be be very well characterized and temporal proximity can be exploited to enable opportunistic loss-limited sampling by changing the size of the windows (within a suggested range) to better identify characteristic signal features.
    For Sliding windows, a user selects the lower and upper bounds of the window and Falkonry determines the appropriate size of the window. A user also has the ability to determine an "assessment rate" that suggests the rate at which Falkonry produces an assessment output.
    In the absence of a user provided assessment rate, Falkonry determines the appropriate assessment rate.



**2. What is a batch Window?**

    In batch window, signal are sampled by splitting them into batch window sizes. This helps improve runtime in scenarios where there is repetition of temporal patterns.
    For batch window a user provide explicit grouping guidance by selecting one of the input signals to serve as a grouping identifier e.g. sample/ batch ID



**3. When should I use Sliding Windows as against batch Windows ?**

     An important interpretation of time series data is whether condition assessments are based on Sliding windows or batch windows. 
     Sliding windows are used when a condition changes on a continuous basis. In some cases, condition assessment is relative to ‘fixed’ window of time. For example, consider a railway switch that intermittently executes ‘throw’ cycles. In such cases we desire to compare one complete throw cycle to another, and are not interested in the long periods between throws. When generating a model for an assessment, you can instruct Falkonry to use either a sliding or a batch  window approach.
     A user can either provide explicit grouping guidance (Batch windows) or provide upper and lower bounds to define a minimum and maximum window width (Sliding windows) which Falkonry will apply to the source data signals.
     In batch windows, signals are sampled by splitting them into fixed/batch window sizes. This helps improve runtime in scenarios where there is repetition of temporal patterns. 
     In Sliding windows, incoming signals may not be be very well characterized and temporal proximity can be exploited to enable opportunistic loss-limited sampling by changing the size of the windows (within a suggested range) to better identify characteristic signal features.



**4. How do I determine the lower and upper bounds when using a Sliding Window approach?**

     In order to set upper and lower bounds on the Sliding window size let’s consider the following example. 
     Consider the signal and 2 supervised model plots shown below. In the signal view at the bottom, there are 3 events (downward spikes). We provide a fact that classifies the first event only. 
     The signal is sampled every 8 mins and the event under consideration (spike) lasts for 3-4 hours on average. The expected failure has a trough (downtime) that lasts ~15-20 mins (or 2 assessment points given the 8 min spread in sampling).  
     When we run a model M[5] (first row) with a bound on Sliding windows from 1 min to 4 hours, the model fails to capture subsequent events. This is because of the lower bound of 1 min fails to capture the entire range of the trough and hence the downward spike. 
     We create another model with a lower bound of 30 min and upper bound of 8 hours and  Falkonry picks up the other two subsequent events based on the one fact provided that classifies this condition. The lower bound of 30 min ensures that there are about 4 assessment points that capture the trough. The upper bound of 8 hours conservatively captures signal characteristics before and after the downward spike.
     The above should hope to serve as a good heuristic in deciding upper and lower bounds on Sliding windows. This example shows how selecting the bounds on Sliding windows helps build the accuracy of the model.


        .. image:: images/sliding_window_bounds.png


**5. How do I determine the bounds for the number of conditions/ states for my model?**

     Falkonry transforms raw signals into meaningful features that differentiate behavior. Clustering is the process of identifying groupings of these feature vectors to characterize historical phenomena. When creating a model in Falkonry, a user has the option to suggest upper and lower bounds on the number of clusters. This gives some control of granularity of the Assessment results.
     Falkonry will try to maximize the number of clusters and hence patterns that be identified based on the signal features. The user can change the bounds of the number of clusters (default being set between 4 and 10) and thus control the impact of the signal feature sets on the number of patterns identified in unsupervised learning.



**6. What is the minimum assessment rate that I can use given different sampling rates of my data?**

     You can specifically choose the "Minimum Interval" option for determining assessment rate when creating a model within Falkonry (refer to FAQ section in :ref:`Assessments<assessments>`
     This option allows assessments as frequently as the input signal data allows. 
     Theoretically, you can choose an assessment rate equal to or slightly greater than the sampling rate of the lowest frequency signal provided that all the signals are aligned in time.



**7. What signals (inputs) do I select when creating a Model?**

     It is essential to pick the right subset of signals to use for model training. Adding unnecessary signals that may not contribute to accurately characterizing a condition/ state may end up deteriorating the results while also increasing run time.
     Avoid signals that are flat over the whole range
     Avoid signals that are monotonously increasing or decreasing over the whole range
     In highly correlated signals select only one in the set

     For the remaining signals it is usually best addressed by the user (SME) to make a judgement in terms of what signals would be necessary for modeling. There can be no better substitutes to domain knowledge and floor experience.

     Falkonry is actively working on addressing this issue. Stay tuned!



**8. What can I do if my Model learning process is frequently failing/aborting?**

     There can be multiple reasons for failing/ aborting model learning processes.
     On private deployments, running more extensive models with multiple high frequency input signals may require additional compute and storage. Increasing the resources being made available to Falkonry often alleviates this problem.
	 For help regarding this, you can send en email to Falkonry Admin at support[at]falkonry[dot]com or you can send in a question or concern using the Intercom which you can find on the top right corner of the UI.



**9. How many models can I create in parallel?** 

     Models can be created simultaneously, given that there are enough hardware resources.
     This number can be configured by your Falkonry Admin (feel free to reach out at support[at]falkonry[dot]com). By default, jumpstarts can create 3 models at the same time.


**10. Is the Falkonry created pattern recognition model dynamic?**

     Falkonry created models are static.  Once created in Falkonry LRS the model remains static.  During live monitoring any new patterns discovered that were not part of model training will be labeled as 'Unknown'.  To classify these newly discovered patterns you have to create a new model revision and redeploy for live monitoring.


**11. Can I rename the machine generated labels (U1, U2, U3, etc.,)?**

     No.  Keeping these labels fixed helps you differentiate between user applied labels for classification vs Falkonry generated labels.


**12. Why can’t I delete the models?**

     Failed and cancelled models can be deleted.  You cannot delete successful models.  The history of models show you the iterations and model parameter adjustments that you tried to arrive at a satisfactory model.


**13. What is the approach to modeling when I have multiple entities in a datastream?**

    Falkonry can train a model on a single entity, a subset or group of entities or all the entities.
    The choice of entities entirely depends on the use case and how differentiated the entities are in terms of characteristics that would help build a more precise model.

    Here are some sample scenarios:

    a] *Single entity based models:* Each entity may behave differently and may need to be modeled separately. For example, similar robotic arms deployed on different production lines building different automobiles.
    In this case the range of movement would be very different and would require modeling each entity by itself.

    b] *Group of entities:* In this case, most of the entities behave similarly but may have some idiosyncractic characteristics that need to be normalized or blended over. 
    For example, gas pipelines with pressure valves measuring gas flow anomalies can largely be modeled as a generic entity with minor structural and environmental differences blended across the single model.

    c] *All entities*: If the entities are all expected to behave in a similar fashion then adding more facts associated with different entities may help improve the accuracy of the model. 
    In such a scenario, it would make sense to create one model (trained over multiple entities) that can be applied across several instances individually. For example, similar elevators in a building that serve the same set of floors.


**14. What should I do when I encounter a “Error in generating output” message?**
    
    Outside of data/fact ingestion errors, model failures can be classified into *model learning* and *model applying* failures.
    Model apply failures result primarily due to hardware resource limitations and constraints.
    
    In the event of a model apply failure and to re-trigger model apply, simply click on the "Regenerate Output" gear icon as shown below

    ..  image:: images/regenerate_output.png

    Clicking on the "Regenerate Output" button would trigger model apply again.
    Note, model apply is also automatically re-triggered when selected a new time range and/or selecting a different list of entities for which model output does not exist.


.. _facts:




Facts
-----

**1. How do I add facts to my model?**

    Facts are known values for Assessments for periods of time in the past. Facts help introduce contextualization needed to add perspective to clusters created by Falkonry and the subsequent step of classification. Supervised models are built on top of these ground truths established from facts.

    There are 4 modes of introducing Facts:

    *1. Manually adding facts from the UI*
    Click on a time segment on a particular model and then click on the blue menu “SELECT EPISODE” and then “Add a Fact” item

        .. image:: images/add_manual_fact_1.png


    
    In the window that opens up provide a fact name and an optional tag and hit “SAVE” or “SAVE AND ADD”.

        .. image:: images/add_manual_fact_2.png



    *2. Falkonry Integration Agents  (e.g. Event Frames from OSIsoft’s PI System)*

    Falkonry’s PI Integration agent for OSIsoft PI System users allows users to import event frames into the Falkonry Service. The integration agent can easily connect to a PI system database using the AFSDK and also allows assessments to be written back into the PI system as attributes. For more details on Falkonry’s PI agent refer to the following link: http://help.falkonry.com/en/latest/pi_agent.html#pi-agent

        .. image:: images/add_piagent_fact.png



    *3. Falkonry supported ADKs*

    Falkonry supports the following development kits that can be used for integrating and extracting data from your choice of data sources:

        `Java <https://github.com/Falkonry/falkonry-java-client>`_

        `C# <https://github.com/Falkonry/falkonry-csharp-client>`_

        `Python <https://github.com/Falkonry/falkonry-python-client>`_


    *4. Uploading csv/ json files with facts data*
	
    Facts can be uploaded, viewed, filtered and downloaded from the UI.  Select “FACTS” button and in the Facts pane click on the gray box which says “Select or Drop CSV or JSON file here”.

    *Sliding window facts*
        CSV:
        <time>		<end>		<entity>	<value>		<tag>

        JSON:
        {"time":X,"end":Y,"entity":"A","value":"B",”tag”:”C”}

    .. image:: images/add_file_facts.png

    *Batch window facts*

        CSV:
        <batch>		<entity>	<value>		<tag>

        JSON:
        {"batch":X,"entity":"A","value":"B",”tag”:”C”}

    Note: In the above examples "tag" is optional. 



**2. How do I delete facts?**

    *1. Facts can be deleted manually from the UI*
    
    Similar to adding facts on the UI, open up the "Select Episode" window and choose "Delete Facts". Drag the mouse over the fact segments and conform deleteion.

        .. image:: images/add_manual_fact_1.png

    *2. Facts can be deleted in batches from the Facts panel*
    
        .. image:: images/facts_button.png

    From the Facts panel, select the "Deletion" radio button and select from any of the following different filters:
        #. Model
        #. Time range
        #. Source
        #. Tags.
        #. Conditions
        #. Entity
        #. Fact upload batch 
        #. Specific facts

    Confirm the fact selections and confirm with the "DELETE" button on the top right of the panel.

    *3. Facts can also be deleted using any one of the ADKs. For details please refer to the following ADK documentation:*

    `Falkonry ADK <http://help.falkonry.com/en/latest/adk_documentation.html>`_    



**3. How do I create various fact subsets?**

    In the presence of multiple facts from different sources (inspection logs, SME generated, historian tags, etc) it may help to group these facts for better classification and fact management.

    Falkonry provides "tags" to help group these signals. These tags can be used to filter facts when creating models, to display facts and to delete fact groups.
    
    You can assign these tags when manually adding facts on the UI.

        .. image:: images/manual_tags.png

    Tags can also be introduced when injecting fact files. Please refer to the CSV, JSON formats shown above. Tags can also be managed through any of the Falkonry ADKs.



.. _support:


Support
--------
    For more details please contact Falkonry Support at <support AT falkonry>


**1. How may I escalate a support issue/ ticket?**

    Reproducible errors that cannot promptly be resolved will be escalated to higher support tiers for further investigation and analysis. 
    Issues will be generally categorized and handled according to an assigned severity level.
    Please contact your customer success manager to esclate the issue.



.. _software_updates:


Services and software updates
------------------------------

**1. How often do you make releases?**

    Falkonry makes software updates every week. These updates are transparent to the users and their installations.

    Falkonry is committed to making the update process as bug-free and easy as possible. Users should expect a pre-release announcement from Falkonry when significant updates are planned to Falkonry LRS.


**2. What is your release strategy?**

    Falkonry makes software updates every week. Jumpstart users on Falkonry cloud environments should see an update 2 days prior to private deployments.
    Falkonry has adopted stringent procedures and methodoligies to test any software updates with a Kanban approach. We rigorously follow our stringent testing procedures for every software update or release before installing on our cloud environments or our customer environments to minimize any impact.


**3. How advance a notice would I get of updates in Shared environment?**

    Falkonry will try and make announcements when significant changes/ updates and improvements are part of the release.
    Any usability changes will be communicated ahead of time to minimize any uncertainities that may hamper user experience.


**4. How often and when does my local instance of Falkonry software gets updated?**

    Falkonry strives to make weekly updates. On certain occasions (significant planned changes) Falkonry may skip a weekly update in lieu of the impact and changes that may need to be tested before releasing to out customers.
    In any case, Falkonry is committed to making the update process as bug-free and easy as possible.


.. _infrastructure:

Infrastructure and Deployment
------------------------------

**1. Can I separate Falkonry components and run them on different nodes/operating systems?**
 
    Yes as long as they are running in same kubernetes cluster.


**2. Does Falkonry support Outbound proxy with authentication?**

    Yes


**3. What operating systems are supported for Falkonry LRS?**

    Ubuntu, CentOS, RedHat


**4. What operating system is required for PI agent?**

    Windows 2015+ server, Windows 7 64 bit 


**5. What types of authentication are supported?**

    #. Google oauth
    #. LinkedIn oauth
    #. auth0
    #. Standard username-password


**6. Can I integrate my local Falkonry instance with my Active Directory?**

    No.  Falkonry will be supporting Active Directory integration in second half of 2018.


**7. What are minimum requirements for a Falkonry LRS Private Environment deployment?**

   To get specific environment requirements for your needs please reach out to Falkonry at support[at]falkonry[dot]com


**8. What are the minimum requirements to run the Falkonry client application?**

    #. Client laptop/desktop should have browser that supports HTML5 (Chrome, Firefox, IE9, Safari)
    #. At least 1GB available for use by the browser
    #. At least 1GHz or better processor
    #. Standard disk/flash based storage used by your organization.
